"""
AI Agent for Bank Statement PDF Parser Generation
Uses LangGraph to create an autonomous agent that generates custom parsers
"""

import os
import sys
import argparse
import subprocess
import pandas as pd
from pathlib import Path
from typing import TypedDict, Literal, Optional
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END

from start import chat_with_llm, client

# ============================================================================
# PYDANTIC MODELS FOR STRUCTURED RESPONSES
# ============================================================================

class PDFAnalysis(BaseModel):
    """Comprehensive analysis of PDF structure and content"""
    text_content: str = Field(description="Extracted text from PDF")
    patterns_found: list[str] = Field(description="Identified patterns like dates, amounts")
    table_structure: str = Field(description="Description of table structure")
    parsing_strategy: str = Field(description="Recommended parsing approach")
    edge_cases_identified: list[str] = Field(description="Specific edge cases and formatting issues found")
    regex_recommendations: list[str] = Field(description="Specific regex patterns recommended for parsing")
    debit_credit_logic: str = Field(description="Detailed logic for determining debit vs credit transactions")
    code_generation_prompt: str = Field(description="Comprehensive prompt for code generation based on analysis")

class ParserCode(BaseModel):
    """Generated parser code"""
    code: str = Field(description="Complete Python code for the parser")
    imports: list[str] = Field(description="Required imports")
    explanation: str = Field(description="Brief explanation of the parsing logic")

class TestResult(BaseModel):
    """Test execution result"""
    passed: bool = Field(description="Whether tests passed")
    achievements: str = Field(description="Achievements of the parser if any")
    error_message: str = Field(description="Error message if test failed")
    suggestions: list[str] = Field(description="Suggestions for fixing issues")

class FixPlan(BaseModel):
    """Plan for fixing parser issues"""
    issues_identified: list[str] = Field(description="Issues found in current parser")
    fix_strategy: str = Field(description="Strategy to fix the issues")
    code_changes: str = Field(description="Specific code changes needed")

# ============================================================================
# STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    """State maintained throughout the agent workflow"""
    target_bank: str
    pdf_path: str
    csv_path: str
    step: int
    analysis: Optional[PDFAnalysis]
    parser_code: Optional[str]
    test_passed: bool
    error_logs: str
    attempt_count: int
    max_attempts: int

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def read_pdf_content(pdf_path: str) -> str:
    """Extract text content from PDF"""
    try:
        import PyPDF2
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    except ImportError:
        # Fallback to pdfplumber if PyPDF2 not available
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
            return text
        except ImportError:
            return "Error: No PDF library available. Install PyPDF2 or pdfplumber."

def create_custom_parser_dir():
    """Create custom_parsers directory if it doesn't exist"""
    parser_dir = Path("custom_parsers")
    parser_dir.mkdir(exist_ok=True)
    
    # Create __init__.py to make it a Python package
    init_file = parser_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text("")

# ============================================================================
# GRAPH NODES
# ============================================================================

def analyze_pdf_node(state: AgentState) -> AgentState:
    """Comprehensively analyze the PDF structure and generate detailed parsing guidance"""
    print(f"Step {state['step']}: Performing comprehensive PDF analysis...")
    
    # Read PDF content
    pdf_content = read_pdf_content(state['pdf_path'])
    
    # Read expected CSV to understand target schema
    expected_df = pd.read_csv(state['csv_path'])
    
    # Extract comprehensive sample for analysis
    lines = pdf_content.split('\n')
    
    # Get different sections of the PDF for comprehensive analysis
    header_lines = '\n'.join(f"Line {i+1}: {repr(line)}" for i, line in enumerate(lines[:5]))
    transaction_sample = '\n'.join(f"Line {i+1}: {repr(line)}" for i, line in enumerate(lines[5:15], 6))
    middle_section = '\n'.join(f"Line {i+1}: {repr(line)}" for i, line in enumerate(lines[25:35], 26))
    end_section = '\n'.join(f"Line {i+1}: {repr(line)}" for i, line in enumerate(lines[-10:], len(lines)-9))
    
    # Count total lines and estimate transactions
    total_lines = len(lines)
    non_empty_lines = len([line for line in lines if line.strip()])
    
    analysis_prompt = f"""
    You are a senior data engineer tasked with performing an EXHAUSTIVE analysis of a bank statement PDF to design a robust parsing system.
    
    DOCUMENT OVERVIEW:
    - Target Bank: {state['target_bank'].upper()}
    - Total lines extracted: {total_lines}
    - Non-empty lines: {non_empty_lines}
    - Expected output transactions: {len(expected_df)}
    
    PDF CONTENT SECTIONS FOR ANALYSIS:
    
    HEADER SECTION:
    {header_lines}
    
    TRANSACTION SAMPLE (Beginning):
    {transaction_sample}
    
    MIDDLE SECTION:
    {middle_section}
    
    END SECTION:
    {end_section}
    
    TARGET OUTPUT SCHEMA:
    Columns: {list(expected_df.columns)}
    Sample expected output:
    {expected_df.head(5).to_string()}
    
    COMPREHENSIVE ANALYSIS REQUIREMENTS:
    
    1. DOCUMENT STRUCTURE ANALYSIS:
    - Identify header patterns, transaction line formats, and footer content
    - Document any page breaks, section separators, or formatting changes
    - Note the exact column order and spacing patterns
    - Identify any metadata lines that should be skipped
    
    2. TRANSACTION LINE PATTERN ANALYSIS:
    - Analyze the exact format: Date | Description | Amount | Balance
    - Document spacing patterns between columns
    - Identify variable-length description fields
    - Note numeric formatting (commas, decimals, negative signs)
    - Identify any lines with trailing text after balance
    
    3. EDGE CASE IDENTIFICATION:
    - Find lines with unusual formatting or extra content
    - Identify transactions that might be missed by simple regex
    - Note any lines containing special keywords like "Karbon"
    - Document potential multi-page boundary issues
    - Identify patterns that could cause false matches
    
    4. REGEX PATTERN RECOMMENDATIONS:
    - Design specific regex patterns for each component
    - Recommend flexible patterns that handle edge cases
    - Suggest patterns that avoid end-of-line anchors for trailing text
    - Provide backup patterns for difficult cases
    
    5. DEBIT/CREDIT CLASSIFICATION LOGIC:
    - Analyze balance change patterns from the samples
    - Design logic for first transaction (no previous balance)
    - Recommend tolerance levels for floating-point comparisons
    - Suggest fallback logic for ambiguous cases
    - Document the complete decision tree
    
    6. CODE GENERATION PROMPT:
    Based on your analysis, create a COMPREHENSIVE, DETAILED prompt that a Python developer can use to generate a robust parser. This prompt should:
    - Include all technical specifications you discovered
    - Provide exact formatting requirements
    - Include all edge cases and how to handle them
    - Specify the complete algorithm for debit/credit classification
    - Include quality assurance checks and validation steps
    - Provide detailed error handling requirements
    
    The code generation prompt should be so detailed that any experienced Python developer can implement a working parser without additional guidance.
    
    CRITICAL REQUIREMENTS:
    - The PDF has 2 pages with approximately 100 transactions total
    - No "Dr", "Cr", or "Balance" text markers exist in the PDF
    - Some transaction lines have extra text after the balance amount
    - The parser must handle multi-page extraction correctly
    - Balance change logic is the only way to determine debit vs credit
    
    Provide your comprehensive analysis covering all these aspects.
    """
    
    analysis = chat_with_llm(
        user_message=analysis_prompt,
        response_format=PDFAnalysis,
        custom_system_prompt="You are a senior data engineering consultant specializing in document parsing and data extraction. Provide the most comprehensive technical analysis possible, considering every detail that could affect parsing accuracy."
    )
    
    print(f"Comprehensive analysis complete.")
    print(f"Strategy: {analysis.parsing_strategy}")
    print(f"Edge cases identified: {len(analysis.edge_cases_identified)}")
    print(f"Regex patterns recommended: {len(analysis.regex_recommendations)}")
    
    return {
        **state,
        "analysis": analysis,
        "step": state["step"] + 1
    }

def generate_parser_node(state: AgentState) -> AgentState:
    """Generate parser code using comprehensive analysis from analyze_pdf_node"""
    print(f"Step {state['step']}: Generating parser code from comprehensive analysis...")
    
    # Use the comprehensive code generation prompt created by analyze_pdf_node
    if state['analysis'] and hasattr(state['analysis'], 'code_generation_prompt'):
        code_prompt = state['analysis'].code_generation_prompt
        print(f"Using elaborate analysis prompt from analyze_pdf_node")
        
        # Enhance the prompt with specific technical requirements from analysis
        enhanced_prompt = f"""
        {code_prompt}
        
        ADDITIONAL TECHNICAL CONSTRAINTS FROM ANALYSIS:
        - Parsing Strategy: {state['analysis'].parsing_strategy}
        - Edge Cases to Handle: {state['analysis'].edge_cases_identified}
        - Recommended Regex Patterns: {state['analysis'].regex_recommendations}
        - Debit/Credit Logic: {state['analysis'].debit_credit_logic}
        
        CRITICAL REQUIREMENTS:
        - Function signature MUST be: def parse(pdf_path: str) -> pd.DataFrame
        - Use pdfplumber for PDF reading (extract from ALL pages)
        - Return DataFrame with columns: ['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance']
        - Use pd.NA for missing debit/credit values
        - Handle the specific PDF format with single amount column
        - Do not add any main function
        - Make sure to add final parse function in the ned in which when pdf path is passed it should give the extracted pandas dataframe
        Generate complete working Python code based on the analysis above.
        """
        
        final_prompt = enhanced_prompt
    else:
        # Fallback prompt if analysis doesn't have detailed prompt
        expected_df = pd.read_csv(state['csv_path'])
        pdf_content = read_pdf_content(state['pdf_path'])
        sample_lines = '\n'.join(pdf_content.split('\n')[2:6])
        
        final_prompt = f"""
        Generate a complete Python parser for {state['target_bank']} bank statements.
        
        TARGET SCHEMA: {expected_df.head(3).to_string()}
        SAMPLE LINES: {sample_lines}
        
        REQUIREMENTS:
        - Extract from ALL PDF pages (2 pages expected, ~100 transactions)
        - Handle lines with trailing text after balance numbers
        - Use balance change logic for debit/credit classification
        - Return DataFrame with columns: ['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance']
        - Handle edge cases and provide error handling
        
        Generate complete working Python code with function signature: def parse(pdf_path: str) -> pd.DataFrame
        """
    
    parser_response = chat_with_llm(
        user_message=final_prompt,
        response_format=ParserCode,
        custom_system_prompt="You are an expert Python developer specializing in PDF parsing and data extraction. Write clean, robust code."
    )
    
    # Create parser file
    create_custom_parser_dir()
    parser_file = f"custom_parsers/{state['target_bank']}_parser.py"
    
    with open(parser_file, 'w') as f:
        f.write(parser_response.code)
    
    print(f"Parser code generated: {parser_file}")
    print(f"Strategy: {parser_response.explanation}")
    
    return {
        **state,
        "parser_code": parser_response.code,
        "step": state["step"] + 1
    }

def test_parser_node(state: AgentState) -> AgentState:
    """Test the generated parser"""
    print(f"Step {state['step']}: Testing parser...")
    
    try:
        # Import the generated parser
        sys.path.insert(0, '.')
        parser_module = f"custom_parsers.{state['target_bank']}_parser"
        
        # Dynamic import
        import importlib
        if parser_module in sys.modules:
            importlib.reload(sys.modules[parser_module])
        
        parser = importlib.import_module(parser_module)
        
        # Test the parser
        result_df = parser.parse(state['pdf_path'])
        expected_df = pd.read_csv(state['csv_path'])
        os.makedirs("test_results", exist_ok=True)
        result_df.to_csv(f"test_results/{state['target_bank']}_test_result{state['attempt_count']}.csv", index=False)
        print(f"Parser output saved to test_results/{state['target_bank']}_test_result{state['attempt_count']}.csv")
        # Compare results - enhanced validation
        if result_df.shape == expected_df.shape and list(result_df.columns) == list(expected_df.columns):
            # Check if actual data values are properly extracted (not all None/empty)
            
            # Check if we have valid numeric data in amount columns
            debit_values = result_df['Debit Amt'].dropna()
            credit_values = result_df['Credit Amt'].dropna()
            balance_values = result_df['Balance'].dropna()
            
            # Check if we have reasonable data distribution
            has_valid_amounts = len(debit_values) > 0 or len(credit_values) > 0
            has_valid_balances = len(balance_values) > 0
            
            # Check if descriptions are properly extracted (not containing numbers)
            descriptions = result_df['Description'].dropna()
            has_clean_descriptions = len(descriptions) > 0 and not all(
                any(char.isdigit() for char in str(desc)[-10:]) for desc in descriptions.head(5)
                if isinstance(desc, str)
            )
            
            # Check for data quality issues
            issues = []
            if not has_valid_amounts:
                issues.append("No valid debit or credit amounts found")
            if not has_valid_balances:
                issues.append("No valid balance values found")
            if not has_clean_descriptions:
                issues.append("Descriptions appear to contain unparsed data")
                
            # Check a sample of actual values against expected
            sample_matches = 0
            for i in range(min(5, len(result_df))):
                result_row = result_df.iloc[i]
                expected_row = expected_df.iloc[i]
                
                # Check if date matches
                if result_row['Date'] == expected_row['Date']:
                    sample_matches += 1
                    
                # Check if balance values are reasonable (within 10% tolerance)
                if (pd.notna(result_row['Balance']) and pd.notna(expected_row['Balance']) and
                    abs(float(result_row['Balance']) - float(expected_row['Balance'])) < 0.01):
                    sample_matches += 1
            
            # Additional strict validation - check actual values against expected
            exact_matches = 0
            value_errors = []
            
            for i in range(min(3, len(result_df))):
                result_row = result_df.iloc[i]
                expected_row = expected_df.iloc[i]
                
                # Check date exact match
                if result_row['Date'] != expected_row['Date']:
                    value_errors.append(f"Row {i}: Date mismatch - got '{result_row['Date']}', expected '{expected_row['Date']}'")
                    
                # Check debit amount
                result_debit = result_row['Debit Amt']
                expected_debit = expected_row['Debit Amt']
                if pd.isna(expected_debit) and not pd.isna(result_debit):
                    value_errors.append(f"Row {i}: Expected no debit amount, but got {result_debit}")
                elif not pd.isna(expected_debit) and (pd.isna(result_debit) or abs(float(result_debit) - float(expected_debit)) > 0.01):
                    value_errors.append(f"Row {i}: Debit amount mismatch - got {result_debit}, expected {expected_debit}")
                    
                # Check credit amount  
                result_credit = result_row['Credit Amt']
                expected_credit = expected_row['Credit Amt']
                if pd.isna(expected_credit) and not pd.isna(result_credit):
                    value_errors.append(f"Row {i}: Expected no credit amount, but got {result_credit}")
                elif not pd.isna(expected_credit) and (pd.isna(result_credit) or abs(float(result_credit) - float(expected_credit)) > 0.01):
                    value_errors.append(f"Row {i}: Credit amount mismatch - got {result_credit}, expected {expected_credit}")
                    
                # Check balance
                result_balance = result_row['Balance']
                expected_balance = expected_row['Balance']
                if pd.isna(result_balance) or abs(float(result_balance) - float(expected_balance)) > 0.01:
                    value_errors.append(f"Row {i}: Balance mismatch - got {result_balance}, expected {expected_balance}")
                    
                # Check description (should be similar)
                result_desc = str(result_row['Description']).strip()
                expected_desc = str(expected_row['Description']).strip()
                if len(result_desc) == 0 or result_desc == 'nan':
                    value_errors.append(f"Row {i}: Empty or invalid description")
                    
                # Count exact matches for this row
                if len(value_errors) == 0:
                    exact_matches += 1

            # Require at least 2 out of 3 rows to match exactly, and no critical issues
            if len(issues) == 0 and len(value_errors) == 0 and exact_matches >= 2:
                print(f"Parser test passed!\nHere is the result:\n{result_df.head(3).to_string()}")
                return {
                    **state,
                    "test_passed": True,
                    "error_logs": "",
                    "step": state["step"] + 1
                }
            else:
                all_errors = issues + value_errors
                error_msg = f"Parser validation failed: {'; '.join(all_errors[:5])}"  # Show first 5 errors
                print(f"Test failed: {error_msg}\nHere is the result:\n{result_df.head(3).to_string()}")
                print(f"Expected first 3 rows:\n{expected_df.head(3).to_string()}")
                return {
                    **state,
                    "test_passed": False,
                    "error_logs": error_msg,
                    "step": state["step"] + 1
                }
        else:
            if result_df.shape[0] != expected_df.shape[0]:
                error_msg = f"Row count mismatch. Expected: {expected_df.shape[0]} transactions, Got: {result_df.shape[0]} transactions. CRITICAL: The PDF has 2 pages and should yield exactly 100 transactions. Check if you're extracting from all pages properly."
            else:
                error_msg = f"Schema mismatch. Expected: {expected_df.shape}, Got: {result_df.shape}"
            print(f"Test failed: {error_msg}\nHere is the result:\n{result_df.head(3).to_string()}")
            return {
                **state,
                "test_passed": False,
                "error_logs": error_msg,
                "step": state["step"] + 1
            }
            
    except Exception as e:
        error_msg = f"Parser execution error: {str(e)}"
        print(f"Test failed: {error_msg}")
        return {
            **state,
            "test_passed": False,
            "error_logs": error_msg,
            "step": state["step"] + 1
        }

def fix_parser_node(state: AgentState) -> AgentState:
    """Fix parser based on test failures"""
    print(f"Step {state['step']}: Fixing parser issues...")
    
    # Get PDF content for context
    pdf_content = read_pdf_content(state['pdf_path'])
    sample_lines = '\n'.join(pdf_content.split('\n')[2:8])  # Show more lines for context
    
    fix_prompt = f"""
    PARSER DEBUGGING - The parser failed with these specific errors:
    {state['error_logs']}
    
    ACTUAL PDF FORMAT (you MUST understand this):
    {sample_lines}
    
    Current parser code:
    {state['parser_code']}
    
    Expected output (first 3 rows):
    {pd.read_csv(state['csv_path']).head(3).to_string()}
    
    ANALYSIS APPROACH:
    1. Identify the root cause of the parsing failure
    2. Check if the issue is with PDF text extraction, regex matching, or logic errors
    3. Verify if all expected transactions are being captured
    4. Ensure debit/credit logic aligns with balance changes
    
    COMMON ISSUES TO CHECK:
    - PDF extraction: Are you reading from all pages? Multi-page PDFs need special handling
    - Regex patterns: Is the pattern too strict or too loose for the actual format?
    - Line filtering: Are you accidentally excluding valid transaction lines?
    - Data type conversion: Are numeric fields being parsed correctly?
    - Debit/credit logic: Does the balance change match the transaction amount?
    - Edge cases: Handle lines with unexpected formatting or extra text
    
    DEBUGGING STRATEGY:
    - Print sample extracted lines to verify PDF text extraction
    - Test regex patterns against actual lines that aren't matching
    - Verify transaction count matches expected (this PDF should have 100 transactions)
    - Check balance change calculations for accuracy
    - Ensure first transaction logic is sound when no previous balance exists
    
    Provide specific fixes to address these errors.
    """
    
    fix_plan = chat_with_llm(
        user_message=fix_prompt,
        response_format=FixPlan,
        custom_system_prompt="You are a debugging expert. Analyze the error and provide specific fixes."
    )
    
    # Generate improved code
    improved_code_prompt = f"""
    Apply these fixes to the parser code:
    
    Issues: {fix_plan.issues_identified}
    Strategy: {fix_plan.fix_strategy}
    Changes: {fix_plan.code_changes}
    
    Current code:
    {state['parser_code']}
    
    Generate the complete fixed code:
    """
    
    improved_parser = chat_with_llm(
        user_message=improved_code_prompt,
        response_format=ParserCode,
        custom_system_prompt="Apply the fixes and generate improved parser code."
    )
    
    # Save improved parser
    parser_file = f"custom_parsers/{state['target_bank']}_parser.py"
    with open(parser_file, 'w') as f:
        f.write(improved_parser.code)
    
    print(f"Applied fixes: {fix_plan.fix_strategy}")
    
    return {
        **state,
        "parser_code": improved_parser.code,
        "attempt_count": state["attempt_count"] + 1,
        "step": state["step"] + 1
    }

# ============================================================================
# CONDITIONAL LOGIC
# ============================================================================

def should_continue(state: AgentState) -> Literal["fix", "end"]:
    """Decide whether to continue fixing or end"""
    if state["test_passed"]:
        return "end"
    elif state["attempt_count"] >= state["max_attempts"]:
        print(f"Max attempts ({state['max_attempts']}) reached. Stopping.")
        return "end"
    else:
        return "fix"

# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def create_agent_graph() -> StateGraph:
    """Create the LangGraph workflow"""
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("analyze", analyze_pdf_node)
    workflow.add_node("generate", generate_parser_node)
    workflow.add_node("test", test_parser_node)
    workflow.add_node("fix", fix_parser_node)
    
    # Set entry point
    workflow.set_entry_point("analyze")
    
    # Add edges
    workflow.add_edge("analyze", "generate")
    workflow.add_edge("generate", "test")
    workflow.add_edge("fix", "generate")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "test",
        should_continue,
        {
            "fix": "fix",
            "end": END
        }
    )
    
    return workflow.compile()

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_final_results(target_bank: str, final_state: AgentState):
    """Provide comprehensive analysis of failed parsing attempts"""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE ANALYSIS & DIAGNOSTICS")
    print("=" * 80)
    
    # Load expected results for comparison
    expected_df = pd.read_csv(f"data/{target_bank}/result.csv")
    expected_count = len(expected_df)
    
    # Try to load the latest parser result if available
    latest_result_file = f"test_results/{target_bank}_test_result{final_state['attempt_count']-1}.csv"
    actual_count = 0
    actual_df = None
    
    try:
        if os.path.exists(latest_result_file):
            actual_df = pd.read_csv(latest_result_file)
            # Filter out empty rows if any
            actual_df = actual_df.dropna(how='all')
            actual_count = len(actual_df)
    except Exception as e:
        print(f"Warning: Could not read result file {latest_result_file}: {e}")
        pass
    
    # Calculate progress metrics
    completion_percentage = (actual_count / expected_count) * 100 if expected_count > 0 else 0
    missing_transactions = expected_count - actual_count
    
    print(f"\nPROGRESS METRICS:")
    print(f"  Expected transactions: {expected_count}")
    print(f"  Successfully extracted: {actual_count}")
    print(f"  Completion rate: {completion_percentage:.1f}%")
    print(f"  Missing transactions: {missing_transactions}")
    
    # Analyze what's working vs not working
    print(f"\nOUTPUT ALIGNMENT ANALYSIS:")
    
    if actual_df is not None and len(actual_df) > 0:
        # Check schema alignment
        expected_columns = list(expected_df.columns)
        actual_columns = list(actual_df.columns)
        schema_match = expected_columns == actual_columns
        
        print(f"  Schema alignment: {'✓ CORRECT' if schema_match else '✗ INCORRECT'}")
        if not schema_match:
            print(f"    Expected: {expected_columns}")
            print(f"    Actual: {actual_columns}")
        
        # Check data quality for available transactions
        if len(actual_df) > 0:
            # Check for valid dates
            valid_dates = actual_df['Date'].str.match(r'\d{2}-\d{2}-\d{4}').sum()
            date_accuracy = (valid_dates / len(actual_df)) * 100
            
            # Check for non-empty descriptions
            valid_descriptions = actual_df['Description'].notna().sum()
            desc_accuracy = (valid_descriptions / len(actual_df)) * 100
            
            # Check for valid amounts (either debit or credit)
            valid_amounts = (actual_df['Debit Amt'].notna() | actual_df['Credit Amt'].notna()).sum()
            amount_accuracy = (valid_amounts / len(actual_df)) * 100
            
            # Check for valid balances
            valid_balances = actual_df['Balance'].notna().sum()
            balance_accuracy = (valid_balances / len(actual_df)) * 100
            
            print(f"  Date format accuracy: {date_accuracy:.1f}%")
            print(f"  Description extraction: {desc_accuracy:.1f}%")
            print(f"  Amount allocation: {amount_accuracy:.1f}%")
            print(f"  Balance extraction: {balance_accuracy:.1f}%")
            
            # Sample comparison (first 3 rows)
            print(f"\nSAMPLE DATA COMPARISON:")
            print(f"  Expected (first 3 rows):")
            for i in range(min(3, len(expected_df))):
                row = expected_df.iloc[i]
                print(f"    {row['Date']} | {row['Description'][:30]}... | D:{row['Debit Amt']} | C:{row['Credit Amt']} | B:{row['Balance']}")
            
            print(f"  Actual (first 3 rows):")
            for i in range(min(3, len(actual_df))):
                row = actual_df.iloc[i]
                print(f"    {row['Date']} | {row['Description'][:30]}... | D:{row['Debit Amt']} | C:{row['Credit Amt']} | B:{row['Balance']}")
    else:
        print(f"  No valid output generated - parser failed completely")
    
    # Identify specific problems
    print(f"\nUNDERLYING PROBLEMS IDENTIFIED:")
    
    problems = []
    if actual_count == 0:
        problems.append("[CRITICAL] No transactions extracted - regex pattern failure")
        problems.append("[CRITICAL] PDF text extraction or line parsing completely failed")
    elif actual_count < expected_count * 0.5:
        problems.append("[MAJOR] Less than 50% extraction rate - regex too restrictive")
        problems.append("[MAJOR] Possible multi-page extraction issues")
    elif actual_count < expected_count * 0.8:
        problems.append("[MODERATE] Missing 20%+ transactions - edge cases not handled")
        problems.append("[MODERATE] Regex pattern missing specific formats")
    else:
        problems.append("[MINOR] High extraction rate - minor edge cases remaining")
    
    # Common issues analysis
    if "Row count mismatch" in final_state['error_logs']:
        problems.append("[ROW COUNT] Check multi-page extraction and line filtering")
    if "could not convert string to float" in final_state['error_logs']:
        problems.append("[DATA TYPE] Regex capturing non-numeric data in amount/balance fields")
    if "Empty DataFrame" in final_state['error_logs']:
        problems.append("[COMPLETE FAILURE] Regex pattern not matching any lines")
    
    for i, problem in enumerate(problems, 1):
        print(f"  {i}. {problem}")
    
    # Human intervention recommendations
    print(f"\nHUMAN INTERVENTION REQUIRED:")
    
    interventions = []
    
    if actual_count == 0:
        interventions.append("1. [DEBUG] Manually test regex patterns against sample PDF lines")
        interventions.append("2. [DEBUG] Verify PDF text extraction is working correctly")
        interventions.append("3. [FIX] Provide working regex template to agent")
    elif actual_count < expected_count * 0.7:
        interventions.append("1. [ANALYZE] Identify which transaction types are being missed")
        interventions.append("2. [ANALYZE] Check for formatting variations in missing transactions")
        interventions.append("3. [ENHANCE] Add specific handling for edge cases")
    else:
        interventions.append("1. [FINE-TUNE] Identify specific lines causing failures")
        interventions.append("2. [FINE-TUNE] Adjust regex to handle remaining edge cases")
    
    # Always recommend these
    interventions.extend([
        f"4. [VALIDATE] Manually compare sample output vs expected for {target_bank}",
        "5. [TEST] Run parser on subset of data to isolate issues",
        "6. [DOCUMENT] Add identified edge cases to agent knowledge base"
    ])
    
    for intervention in interventions:
        print(f"  {intervention}")
    
    # Specific recommendations based on error patterns
    print(f"\nSPECIFIC RECOMMENDATIONS:")
    
    if "4047.68ChatGPT" in str(actual_df) if actual_df is not None else False:
        print("  * [EDGE CASE] Detected line with extra text after balance")
        print("    -> Remove $ anchor from regex pattern")
        print("    -> Add handling for trailing text in balance field")
    
    if actual_count > 0 and actual_count < expected_count:
        print("  * [PARTIAL] Partial extraction detected")
        print("    -> Check if first/last transactions are being skipped")
        print("    -> Verify multi-page extraction logic")
        print("    -> Review line filtering conditions")
    
    if completion_percentage > 90:
        print("  * [HIGH SUCCESS] High accuracy achieved - minimal manual fixes needed")
        print("    -> Focus on identifying the few remaining edge cases")
        print("    -> Consider this an acceptable result for production")
    elif completion_percentage > 70:
        print("  * [MODERATE SUCCESS] Systematic improvement possible")
        print("    -> Pattern analysis of missing transactions recommended")
        print("    -> Agent learning could be enhanced with better examples")
    else:
        print("  * [LOW SUCCESS] Fundamental approach issues")
        print("    -> Consider providing more explicit parsing templates")
        print("    -> Manual debugging of core parsing logic required")
    
    print("\n" + "=" * 80)

# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="AI Agent for Bank Statement PDF Parser Generation")
    parser.add_argument("--target", required=True, help="Target bank (e.g., icici, sbi)")
    
    args = parser.parse_args()
    
    # Validate input files
    pdf_path = f"data/{args.target}/{args.target} sample.pdf"  # Handle space in filename
    csv_path = f"data/{args.target}/result.csv"
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)
    
    print(f"Starting AI Agent for {args.target.upper()} parser generation...")
    print(f"PDF: {pdf_path}")
    print(f"Expected CSV: {csv_path}")
    print("-" * 60)
    
    # Initialize state
    initial_state = {
        "target_bank": args.target,
        "pdf_path": pdf_path,
        "csv_path": csv_path,
        "step": 1,
        "analysis": None,
        "parser_code": None,
        "test_passed": False,
        "error_logs": "",
        "attempt_count": 1,
        "max_attempts": 3
    }
    
    # Create and run the agent
    agent_graph = create_agent_graph()
    
    try:
        final_state = agent_graph.invoke(initial_state)
        
        print("-" * 60)
        if final_state["test_passed"]:
            print(f"SUCCESS! Parser for {args.target.upper()} generated successfully!")
            print(f"Parser saved to: custom_parsers/{args.target}_parser.py")
        else:
            print(f"FAILED: Could not generate working parser after {final_state['attempt_count']} attempts")
            print(f"Last error: {final_state['error_logs']}")
            
            # Generate comprehensive analysis
            analyze_final_results(args.target, final_state)
        
    except Exception as e:
        print(f"Agent execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()