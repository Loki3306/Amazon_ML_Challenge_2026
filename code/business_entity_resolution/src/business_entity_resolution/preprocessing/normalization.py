import polars as pl

def get_normalization_exprs(
    name_col: str = "business_name",
    addr_col: str = "business_address",
    country_col: str = "country"
) -> list[pl.Expr]:
    """
    Returns a list of Polars expressions to normalize the entity fields.
    This uses native Polars string operations for maximum performance (streaming/batching).
    
    Normalization steps applied:
    1. Lowercase
    2. Replace punctuation with space
    3. Collapse multiple whitespaces into a single space
    4. Strip leading/trailing whitespaces
    5. Handle nulls by keeping them as null or filling with empty string (keeping nulls for now to preserve raw semantics, though empty strings are safer for text algorithms. We will fill nulls with empty string for the _norm columns).
    """
    
    # Regex to match most common punctuation: ! " # $ % & ' ( ) * + , - . / : ; < = > ? @ [ \ ] ^ _ ` { | } ~
    # We use a broad punctuation regex class `\p{P}` if supported, or explicitly common ones.
    # Polars uses the regex crate (Rust), so `\p{P}` works perfectly.
    punct_pattern = r"\p{P}"
    whitespace_pattern = r"\s+"
    
    exprs = []
    
    if name_col:
        exprs.append(
            pl.col(name_col)
            .fill_null("")
            .str.to_lowercase()
            .str.replace_all(punct_pattern, " ")
            .str.replace_all(whitespace_pattern, " ")
            .str.strip_chars()
            .alias(f"name_norm")
        )
        
    if addr_col:
        exprs.append(
            pl.col(addr_col)
            .fill_null("")
            .str.to_lowercase()
            .str.replace_all(punct_pattern, " ")
            .str.replace_all(whitespace_pattern, " ")
            .str.strip_chars()
            .alias(f"address_norm")
        )
        
    if country_col:
        exprs.append(
            pl.col(country_col)
            .fill_null("")
            .str.to_lowercase()
            .str.replace_all(punct_pattern, " ")
            .str.replace_all(whitespace_pattern, " ")
            .str.strip_chars()
            .alias(f"country_norm")
        )
        
    return exprs
