from __future__ import annotations

from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED
from html import escape
import math
import pandas as pd
import numpy as np


def _clean_sheet_name(name: str) -> str:
    bad = ['\\', '/', '?', '*', '[', ']', ':']
    safe = str(name)[:31]
    for char in bad:
        safe = safe.replace(char, '-')
    return safe or 'Sheet'


def _cell_ref(row: int, col: int) -> str:
    letters = ''
    while col:
        col, rem = divmod(col - 1, 26)
        letters = chr(65 + rem) + letters
    return f'{letters}{row}'


def _is_number(value) -> bool:
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return False


def _format_value(value) -> str:
    if value is None:
        return ''
    try:
        if pd.isna(value):
            return ''
    except Exception:
        pass
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime('%Y-%m-%d')
    return str(value)


def _df_to_rows(df: pd.DataFrame, include_index: bool = True) -> list[list]:
    if df is None:
        return []
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.copy()

    rows: list[list] = []
    if include_index:
        index_name = df.index.name or 'Date'
        rows.append([index_name] + list(df.columns))
        for idx, row in df.iterrows():
            rows.append([idx] + row.tolist())
    else:
        rows.append(list(df.columns))
        rows.extend(df.astype(object).values.tolist())
    return rows


def _sheet_xml(rows: list[list]) -> str:
    xml_rows = []
    for r_idx, row in enumerate(rows, start=1):
        cells = []
        for c_idx, value in enumerate(row, start=1):
            ref = _cell_ref(r_idx, c_idx)
            if _is_number(value):
                cells.append(f'<c r="{ref}"><v>{float(value)}</v></c>')
            else:
                text = escape(_format_value(value))
                cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>')
        xml_rows.append(f'<row r="{r_idx}">' + ''.join(cells) + '</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetData>' + ''.join(xml_rows) + '</sheetData>'
        '</worksheet>'
    )


def _workbook_xml(sheet_names: list[str]) -> str:
    sheets = ''.join(
        f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>'
        for i, name in enumerate(sheet_names, start=1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets>' + sheets + '</sheets></workbook>'
    )


def _workbook_rels_xml(sheet_count: int) -> str:
    rels = ''.join(
        f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>'
        for i in range(1, sheet_count + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + rels + '</Relationships>'
    )


def _content_types_xml(sheet_count: int) -> str:
    overrides = ''.join(
        f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for i in range(1, sheet_count + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        + overrides + '</Types>'
    )


def _root_rels_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '</Relationships>'
    )


def _build_xlsx(sheets: dict[str, pd.DataFrame]) -> bytes:
    output = BytesIO()
    sheet_names = []
    used = set()
    for name in sheets.keys():
        base = _clean_sheet_name(name)
        final = base
        n = 2
        while final in used:
            suffix = f'_{n}'
            final = base[:31 - len(suffix)] + suffix
            n += 1
        used.add(final)
        sheet_names.append(final)

    with ZipFile(output, 'w', ZIP_DEFLATED) as zf:
        zf.writestr('[Content_Types].xml', _content_types_xml(len(sheet_names)))
        zf.writestr('_rels/.rels', _root_rels_xml())
        zf.writestr('xl/workbook.xml', _workbook_xml(sheet_names))
        zf.writestr('xl/_rels/workbook.xml.rels', _workbook_rels_xml(len(sheet_names)))
        for i, (df_name, df) in enumerate(sheets.items(), start=1):
            include_index = df_name not in {'Metrics', 'Trade Log'}
            rows = _df_to_rows(df, include_index=include_index)
            zf.writestr(f'xl/worksheets/sheet{i}.xml', _sheet_xml(rows))
    output.seek(0)
    return output.getvalue()


def excel_report(
    metrics: dict,
    returns: pd.Series,
    equity: pd.Series,
    trades: pd.DataFrame | None = None,
    extra_sheets: dict[str, pd.DataFrame] | None = None,
) -> bytes:
    """Generate an .xlsx report without openpyxl/xlsxwriter.

    The file is a valid lightweight XLSX package built with Python's standard
    library. This avoids dependency issues on Python 3.14/macOS.
    """
    sheets: dict[str, pd.DataFrame] = {
        'Metrics': pd.DataFrame(metrics.items(), columns=['Metric', 'Value']),
        'Equity Curve': equity.rename('Equity').to_frame(),
        'Daily Returns': returns.rename('Returns').to_frame(),
    }
    if trades is not None and not trades.empty:
        sheets['Trade Log'] = trades
    if extra_sheets:
        for name, df in extra_sheets.items():
            sheets[_clean_sheet_name(name)] = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
    return _build_xlsx(sheets)
