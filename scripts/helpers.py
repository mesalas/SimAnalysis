def get_column_colors(df):
    colors = list()
    for column_name in df.columns:
        if "ST" in column_name:
            colors.append("orange")
        elif "Fast" in column_name:
            colors.append("red")
        else:
            colors.append("steelblue")
    return colors