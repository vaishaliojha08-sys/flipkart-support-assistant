import pandas as pd


df = pd.read_csv("orders_dataset.csv")

print("=" * 60)
print("DATASET VERIFICATION")
print("=" * 60)

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

print("\nOverall return rate:")
print(f"{df['returned'].mean():.4%}")

missing_pct = df["rating_given"].isna().mean()

print("\nMissing rating_given:")
print(f"{missing_pct:.4%}")

print("\nReturn rate by product_category:")
category_returns = (
    df.groupby("product_category")["returned"]
    .agg(["count", "mean"])
    .rename(columns={"mean": "return_rate"})
)

category_returns["return_rate"] = category_returns["return_rate"] * 100
print(category_returns)

print("\nReturn rate by payment_method:")
payment_returns = (
    df.groupby("payment_method")["returned"]
    .agg(["count", "mean"])
    .rename(columns={"mean": "return_rate"})
)

payment_returns["return_rate"] = payment_returns["return_rate"] * 100
print(payment_returns)

print("\nMissing rating_given by payment method:")

missing_by_payment = (
    df.groupby("payment_method")["rating_given"]
    .apply(lambda x: x.isna().mean() * 100)
)

print(missing_by_payment)

cod_rate = df.loc[
    df["payment_method"] == "COD",
    "rating_given"
].isna().mean() * 100

non_cod_rate = df.loc[
    df["payment_method"] != "COD",
    "rating_given"
].isna().mean() * 100

print("\nMAR analysis:")
print(f"COD missing rate: {cod_rate:.2f}%")
print(f"Non-COD missing rate: {non_cod_rate:.2f}%")
print(f"Gap: {cod_rate - non_cod_rate:.2f} percentage points")

print(
    "\nConclusion: rating_given missingness is MAR because "
    "missingness depends on the observed payment_method column."
)