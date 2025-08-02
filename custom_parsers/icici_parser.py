import pandas as pd
import numpy as np
import re
import pdfplumber


def preprocess_text(text):
    lines = text.split('\n')
    lines = [line.strip() for line in lines]
    
    # Find the index of the first line that matches the date pattern to skip header
    header_end_index = 0
    for i, line in enumerate(lines):
        if re.match(r'\d{2}-\d{2}-\d{4}', line):
            header_end_index = i
            break
    
    lines = lines[header_end_index:]
    lines = [line for line in lines if line]
    return lines


def extract_transaction_data(line):
    pattern = r'^(\d{2}-\d{2}-\d{4})\s+(.*?)\s+(?:(\d+\.?\d*)\s+)?(?:(\d+\.?\d*)\s+)?(-?\d+\.?\d*)(?:\s+.*)?$'
    match = re.match(pattern, line)
    if match:
        date_str, description, debit_str, credit_str, balance_str = match.groups()
        
        debit = float(debit_str) if debit_str else None
        credit = float(credit_str) if credit_str else None
        balance = float(balance_str)
        return date_str, description, debit, credit, balance
    else:
        return None


def classify_debit_credit(balance, previous_balance):
    if previous_balance is None:
        return np.nan, np.nan
    else:
        balance_difference = balance - previous_balance
        if balance_difference > 0:
            return np.nan, abs(balance_difference)
        elif balance_difference < 0:
            return abs(balance_difference), np.nan
        else:
            print("Warning: Ambiguous transaction with no balance change.")
            return np.nan, np.nan


def validate_data(df):
    # Data Type Validation
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    for col in ['Debit Amt', 'Credit Amt', 'Balance']:
        df[col] = pd.to_numeric(df[col], errors='raise')
    
    # Missing Value Check (critical columns)
    for col in ['Date', 'Description', 'Balance']:
        if df[col].isnull().any():
            print(f"Warning: Missing values found in column '{col}'.")

    # Basic amount check (example - adjust as needed)
    for col in ['Debit Amt', 'Credit Amt', 'Balance']:
        if not df[col].isnull().all():  # Skip if the entire column is NaN
            if (df[col].abs() > 1e9).any(): #Check for unreasonably large amounts
                print(f'Warning: Unusually high values found in column {col}')


def parse_icici_statement(text):
    lines = preprocess_text(text)
    
    data = []
    previous_balance = None
    
    for i, line in enumerate(lines):
        transaction_data = extract_transaction_data(line)
        if transaction_data:
            date_str, description, debit, credit, balance = transaction_data
            
            if debit is not None and credit is not None:
                pass # Debit and credit are parsed directly
            else:
                # Determine debit/credit based on balance change
                if previous_balance is None:
                    debit, credit = np.nan, np.nan #First transaction
                else:
                    if debit is None and credit is None:  # Amounts not directly available
                        debit, credit = classify_debit_credit(balance, previous_balance)
                    elif debit is None:
                         debit = np.nan
                    elif credit is None:
                         credit = np.nan

            data.append([date_str, description, debit, credit, balance])
            previous_balance = balance
        else:
            print(f"Error: Could not parse line {i + 1}: {line}")

    df = pd.DataFrame(data, columns=['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance'])
    
    # Convert to correct types and handle missing values
    df = df.replace({np.nan: pd.NA})  # Replace numpy nan with pandas NA
    try:
        validate_data(df)
    except Exception as e:
        print(f"Validation Error: {e}")

    return df


import pdfplumber

def parse(pdf_path: str) -> pd.DataFrame:
    text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return parse_icici_statement(text)
