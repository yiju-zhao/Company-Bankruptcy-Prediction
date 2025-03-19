# data_processing.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: f'{x:,.4f}')

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads the American Bankruptcy CSV dataset from a specified file path.
    
    Parameters:
        file_path (str): The path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    df = pd.read_csv(file_path)
    return df

def initial_exploration(df: pd.DataFrame) -> None:
    """
    Prints basic information about the dataset:
      - Shape
      - Column names
      - Data types
      - Missing values
      - Summary statistics (numerical)
    
    Parameters:
        df (pd.DataFrame): The loaded dataset.
    """
    print("=== INITIAL EXPLORATION ===")
    print(f"Data Shape: {df.shape}\n")
    
    print("Column Names:")
    print(df.columns.tolist(), "\n")
    
    print("Data Types:")
    display(df.dtypes)
    
    # Number of unique companies
    if 'company_name' in df.columns:
        num_unique_companies = df['company_name'].nunique()
        print(f"\nNumber of unique companies: {num_unique_companies}")
    
    # Missing values
    missing_vals = df.isnull().sum()
    print("\nMissing values per column:")
    print(missing_vals)
    
    # Summary stats
    numeric_cols = df.select_dtypes(include=np.number).columns
    if not numeric_cols.empty:
        print("\nStatistical Summary (Numeric Columns):")
        display(df[numeric_cols].describe())

def distribution_and_correlation(df: pd.DataFrame) -> None:
    """
    Plots histograms for numeric columns and a correlation matrix.
    
    Parameters:
        df (pd.DataFrame): The dataset.
    """
    print("\n=== DISTRIBUTIONS & CORRELATION ===")
    numeric_cols = df.select_dtypes(include=np.number).columns
    
    # Histograms
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        plt.hist(df[col].dropna(), bins=30, edgecolor='black')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()
    
    # Correlation matrix
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        plt.figure(figsize=(10, 8))
        plt.matshow(corr_matrix, fignum=1)
        plt.colorbar()
        plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
        plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
        plt.title("Correlation Matrix", pad=20)
        plt.show()

def year_based_analysis(df: pd.DataFrame) -> None:
    """
    Analyzes the number of companies per year, identifies left-censored
    companies, and shows new entries by year (>= 1999).
    
    Parameters:
        df (pd.DataFrame): The dataset.
    """
    print("\n=== YEAR-BASED ANALYSIS ===")
    if 'company_name' not in df.columns or 'year' not in df.columns:
        print("Required columns ('company_name' or 'year') not found.")
        return
    
    # Sort for chronological order
    df.sort_values(by=['company_name', 'year'], inplace=True)
    
    # Companies by year
    companies_by_year = df.groupby('year')['company_name'].nunique()
    plt.figure(figsize=(12, 6))
    companies_by_year.plot(kind='bar')
    plt.xlabel('Year')
    plt.ylabel('Number of Companies')
    plt.title('Number of Companies by Year')
    plt.xticks(rotation=45)
    plt.show()
    
    # First year each company appears
    first_year = df.groupby('company_name')['year'].min().reset_index()
    first_year.columns = ['company_name', 'first_year']
    
    # Left-censored = first appearance in 1999
    left_censored_companies = first_year[first_year['first_year'] == 1999]
    num_left_censored_companies = left_censored_companies['company_name'].nunique()
    print(f"Number of left-censored companies (first year = 1999): {num_left_censored_companies}")
    
    # New entries by year (>= 1999)
    first_year_after_1999 = first_year[first_year['first_year'] >= 1999]
    new_entries_by_year = first_year_after_1999['first_year'].value_counts().sort_index()
    
    plt.figure(figsize=(12, 6))
    new_entries_by_year.plot(kind='bar')
    plt.xlabel('Year')
    plt.ylabel('Number of New Entries')
    plt.title('New Entries by Year (>= 1999)')
    plt.xticks(rotation=45)
    plt.show()

def failed_status_analysis(df: pd.DataFrame) -> None:
    """
    Analyzes the 'failed' status by year, multiple failures, 
    first vs. last failure events, and checks if any companies 
    are alive after failing.
    
    Parameters:
        df (pd.DataFrame): The dataset.
    """
    print("\n=== 'FAILED' STATUS ANALYSIS ===")
    if 'status_label' not in df.columns or 'company_name' not in df.columns or 'year' not in df.columns:
        print("Required columns ('status_label', 'company_name', 'year') not found.")
        return
    
    # Simple count of failed statuses by year
    failed_df_simple = df[df['status_label'] == 'failed']
    failed_by_year_simple = failed_df_simple['year'].value_counts().sort_index()
    
    plt.figure(figsize=(12, 6))
    failed_by_year_simple.plot(kind='bar')
    plt.xlabel('Year')
    plt.ylabel('Number of Failed Statuses')
    plt.title('Number of Failed Statuses by Year')
    plt.xticks(rotation=45)
    plt.show()
    
    # Multiple failed statuses
    failed_counts = failed_df_simple.groupby('company_name').size()
    companies_with_multiple_failures = failed_counts[failed_counts > 1]
    print(f"Number of companies with more than one 'failed' status: {companies_with_multiple_failures.count()}")
    print(f"Max number of 'failed' statuses for a single company: {failed_counts.max()}")
    
    # Merge first_failed year
    df.sort_values(by=['company_name', 'year'], inplace=True)
    first_failed = df[df['status_label'] == 'failed'].groupby('company_name')['year'].min().reset_index()
    first_failed.columns = ['company_name', 'first_failed_year']
    df_merged = df.merge(first_failed, on='company_name', how='left')
    
    # Observations to drop if ignoring post-first-failure
    observations_to_drop = df_merged[
        (df_merged['status_label'] == 'failed') & 
        (df_merged['year'] > df_merged['first_failed_year'])
    ]
    print(f"Number of observations lost post-first-failure: {observations_to_drop.shape[0]}")
    
    # First failure only
    failed_first_only = failed_df_simple.drop_duplicates(subset='company_name', keep='first')
    failed_by_year_first = failed_first_only['year'].value_counts().sort_index()
    
    plt.figure(figsize=(12, 6))
    failed_by_year_first.plot(kind='bar')
    plt.xlabel('Year')
    plt.ylabel('Number of Failed Statuses')
    plt.title('Failed Statuses by Year (First Failure Only)')
    plt.xticks(rotation=45)
    plt.show()
    
    # Last failure only
    failed_last_only = failed_df_simple.drop_duplicates(subset='company_name', keep='last')
    failed_by_year_last = failed_last_only['year'].value_counts().sort_index()
    
    plt.figure(figsize=(12, 6))
    failed_by_year_last.plot(kind='bar')
    plt.xlabel('Year')
    plt.ylabel('Number of Failed Statuses')
    plt.title('Failed Statuses by Year (Last Failure Only)')
    plt.xticks(rotation=45)
    plt.show()
    
    # Compare original vs. deduplicated
    total_failed_original = failed_df_simple.shape[0]
    total_failed_first = failed_first_only.shape[0]
    total_failed_last = failed_last_only.shape[0]
    print(f"Total 'failed' (original): {total_failed_original}")
    print(f"Total 'failed' (first instance): {total_failed_first}")
    print(f"Total 'failed' (last instance): {total_failed_last}")
    
    # Alive after failed
    companies_with_failed = df[df['status_label'] == 'failed']['company_name'].unique()
    filtered_df = df[df['company_name'].isin(companies_with_failed)].sort_values(by=['company_name', 'year'])
    
    num_companies_with_alive_after_failed = 0
    for _, company_df in filtered_df.groupby('company_name'):
        failed_years = company_df[company_df['status_label'] == 'failed']['year']
        if not failed_years.empty:
            earliest_fail_year = failed_years.min()
            # Check if there's an 'alive' after earliest fail year
            alive_after_fail = company_df[
                (company_df['status_label'] == 'alive') & 
                (company_df['year'] > earliest_fail_year)
            ]
            if not alive_after_fail.empty:
                num_companies_with_alive_after_failed += 1
    
    print(f"Companies with 'alive' status after any 'failed' status: {num_companies_with_alive_after_failed}")

def additional_eda(df: pd.DataFrame) -> None:
    """
    Demonstrates additional EDA steps:
      - Numeric features by status (describe, histograms)
      - Crosstab of status by year
      - Failure rate over time (unique companies)
      - Correlation with binary failure
      - Duplicate (company, year) checks
    """
    print("\n=== ADDITIONAL EDA ===")
    
    # 1) Describe numeric features by status
    numeric_cols = df.select_dtypes(include=np.number).columns
    if 'status_label' in df.columns:
        print("Descriptive Stats by Status (Numeric Columns):")
        display(df.groupby('status_label')[numeric_cols].describe())
    
    # 2) Histograms by status
    if 'status_label' in df.columns:
        statuses = df['status_label'].unique()
        for col in numeric_cols:
            plt.figure(figsize=(8, 4))
            for status in statuses:
                subset = df[df['status_label'] == status]
                plt.hist(subset[col].dropna(), bins=30, alpha=0.5, label=status)
            plt.title(f"Distribution of '{col}' by Status")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.legend()
            plt.show()
    
    # 3) Crosstab of status by year
    if {'year', 'status_label'}.issubset(df.columns):
        status_year_ct = pd.crosstab(df['year'], df['status_label'])
        print("\nCounts of Status by Year:")
        display(status_year_ct)
        
        plt.figure(figsize=(10, 6))
        status_year_ct.plot(kind='bar', stacked=False)
        plt.title("Counts of Status by Year")
        plt.xlabel("Year")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    # 4) Failure rate over time
    if {'year', 'company_name', 'status_label'}.issubset(df.columns):
        companies_by_year = df.groupby('year')['company_name'].nunique()
        failed_by_year = df[df['status_label'] == 'failed'].groupby('year')['company_name'].nunique()
        
        failure_rate_df = pd.DataFrame({
            'total_companies': companies_by_year,
            'failed_companies': failed_by_year
        }).fillna(0)
        failure_rate_df['fail_rate'] = (failure_rate_df['failed_companies'] /
                                        failure_rate_df['total_companies'])
        
        print("\nFailure Rate by Year (Unique Companies):")
        display(failure_rate_df)
        
        plt.figure(figsize=(8, 4))
        plt.plot(failure_rate_df.index, failure_rate_df['fail_rate'], marker='o')
        plt.title("Failure Rate Over Time")
        plt.xlabel("Year")
        plt.ylabel("Failure Rate (Failed / Total)")
        plt.xticks(rotation=45)
        plt.show()
    
    # 5) Correlation with a binary 'failed' indicator
    if 'status_label' in df.columns:
        df['failed_binary'] = df['status_label'].map({'failed': 1, 'alive': 0})
        corr_list = {}
        for col in numeric_cols:
            valid_mask = df[col].notnull() & df['failed_binary'].notnull()
            if valid_mask.sum() > 1:
                corr_value = np.corrcoef(
                    df.loc[valid_mask, col],
                    df.loc[valid_mask, 'failed_binary']
                )[0, 1]
                corr_list[col] = corr_value
        
        if corr_list:
            sorted_corr = dict(sorted(corr_list.items(), key=lambda x: abs(x[1]), reverse=True))
            print("\nCorrelation with 'failed_binary' (1=Failed, 0=Alive):")
            for feature, cval in sorted_corr.items():
                print(f"{feature}: {cval:.3f}")
    
    # 6) Check duplicates (company_name, year)
    if {'company_name', 'year'}.issubset(df.columns):
        duplicates = df.duplicated(subset=['company_name', 'year'], keep=False)
        if duplicates.any():
            print("\nDUPLICATE (company_name, year) ROWS FOUND:")
            display(df[duplicates].sort_values(['company_name', 'year']))
        else:
            print("\nNo duplicate (company_name, year) rows found.")

def main(file_path: str) -> None:
    """
    Orchestrates the entire data loading and processing workflow
    by calling each function in sequence.
    
    Parameters:
        file_path (str): The path to the CSV file to load and analyze.
    """
    # 1. Load data
    df = load_data(file_path)
    
    # 2. Perform initial exploration
    initial_exploration(df)
    
    # 3. Distribution & Correlation
    distribution_and_correlation(df)
    
    # 4. Year-based analysis
    year_based_analysis(df)
    
    # 5. Failed status analysis
    failed_status_analysis(df)
    
    # 6. Additional EDA
    additional_eda(df)

# -------------------------------------------------------------------
# If you want to run this as a script from the command line, you can
# add the following block. For example:
#
# python data_processing.py
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Replace the path below with your actual file path or a default path
    csv_path = r"C:\Users\Sima\Company-Bankruptcy-Prediction\american_bankruptcy.csv"
    main(csv_path)
