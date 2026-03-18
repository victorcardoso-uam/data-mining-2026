import os
import datetime
import pandas as pd


def _df_to_md(obj):
    try:
        return obj.to_markdown()
    except Exception:
        return obj.to_string()


def build_report(df: pd.DataFrame) -> str:
    lines = []
    lines.append("# Validation report")
    lines.append("")
    lines.append(f"Generated: {datetime.datetime.now().isoformat()}")
    lines.append("")

    lines.append("## 1) DATASET OVERVIEW")
    lines.append(f"- Shape (rows, columns): {df.shape}")
    lines.append("")
    lines.append("**Columns:**")
    lines.append("")
    lines.append(", ".join(list(df.columns)))
    lines.append("")
    lines.append("**Data types:**")
    lines.append("")
    lines.append("```")
    lines.append(df.dtypes.to_string())
    lines.append("```")
    lines.append("")

    lines.append("## 2) MISSING VALUES")
    missing_count = df.isna().sum().sort_values(ascending=False)
    missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    missing_summary = pd.DataFrame({
        "missing_count": missing_count,
        "missing_percentage": missing_pct
    })
    lines.append(_df_to_md(missing_summary.head(10)))
    lines.append("")

    lines.append("## 3) DESCRIPTIVE STATISTICS")
    lines.append(_df_to_md(df.describe()))
    lines.append("")

    lines.append("## 4) DUPLICATE ROWS")
    lines.append(f"- Number of duplicate rows: {df.duplicated().sum()}")
    lines.append("")

    lines.append("## 5) INTEGRITY CHECKS (CUSTOMIZE)")

    if 'SPEEDLIMIT' in df.columns:
        neg_count = int((df['SPEEDLIMIT'] < 0).sum())
        lines.append(f"- Negative speed limits: {neg_count}")
        avg_speed = df['SPEEDLIMIT'].mean()
        lines.append(f"- Average speed limit: {avg_speed:.2f}")
    else:
        lines.append("- Column `SPEEDLIMIT` not found in dataset")

    if 'STATUS' in df.columns:
        uniq = df['STATUS'].dropna().unique().tolist()
        lines.append(f"- Unique values in `STATUS`: {uniq}")
    else:
        lines.append("- Column `STATUS` not found in dataset")

    if 'ARTCLASS' in df.columns:
        uniq_artclass = df['ARTCLASS'].dropna().unique().tolist()
        lines.append(f"- Unique street classes: {uniq_artclass}")
    else:
        lines.append("- Column `ARTCLASS` not found in dataset")

    return "\n\n".join(lines)


def main():
    input_path = os.path.join("data", "raw", "Seattle_Streets_1_-5073353257610679043.csv")
    df = pd.read_csv(input_path)
    report = build_report(df)
    os.makedirs("results", exist_ok=True)
    report_path = os.path.join("results", "session5_validation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()

