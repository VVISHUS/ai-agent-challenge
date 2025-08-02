# AI Agent for Bank Statement PDF Parser Generation

An autonomous AI agent built with LangGraph that automatically generates custom PDF parsers for bank statements. The agent analyzes PDF structure, generates parsing code, tests it, and iteratively improves until achieving accurate extraction.

## 🚀 Features

- **Autonomous PDF Analysis**: Comprehensive analysis of bank statement PDF structure and patterns
- **Code Generation**: Automatically generates custom Python parsers using LLM
- **Iterative Testing**: Tests generated parsers and fixes issues automatically
- **Multi-Bank Support**: Extensible to support different bank statement formats
- **Edge Case Handling**: Identifies and handles complex parsing scenarios
- **Quality Validation**: Ensures extracted data matches expected schema and values

## 📋 Requirements

```
langgraph>=0.2.0
langchain>=0.3.0
langchain-openai>=0.2.0
python-dotenv>=1.0.0
pydantic>=2.0.0
pandas
PyPDF2 or pdfplumber
```

## 🛠️ Installation

1. Clone the repository:
```bash
gh repo clone VVISHUS/ai-agent-challenge
cd ai-agent-challenge
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with:
```
primary_url=your_llm_endpoint_url
GEMINI_API_KEY=your_api_key
```

## 🏃‍♂️ Usage

### Basic Usage

```bash
python agent.py --target icici
```

### Command Line Arguments

- `--target`: Target bank name (required)
  - Example: `icici`, `sbi`, etc.

### File Structure Requirements

Your project should follow this structure:
```
├── data/
│   └── {bank_name}/
│       ├── {bank_name} sample.pdf
│       └── result.csv
├── custom_parsers/
│   └── (generated parsers will be saved here)
└── test_results/
    └── (test outputs will be saved here)
```

## 🔄 Agent Workflow

The agent follows a structured LangGraph workflow:

1. **PDF Analysis** (`analyze_pdf_node`)
   - Extracts text from PDF using PyPDF2/pdfplumber
   - Analyzes document structure and patterns
   - Identifies edge cases and formatting variations
   - Generates comprehensive parsing strategy

2. **Code Generation** (`generate_parser_node`)
   - Creates custom Python parser based on analysis
   - Implements regex patterns for data extraction
   - Adds debit/credit classification logic
   - Handles multi-page extraction

3. **Testing** (`test_parser_node`)
   - Executes generated parser on sample PDF
   - Validates output against expected results
   - Performs strict data quality checks
   - Compares schema and sample values

4. **Iterative Fixing** (`fix_parser_node`)
   - Identifies specific parsing failures
   - Generates targeted fixes
   - Re-implements improved parser
   - Repeats until success or max attempts

## 📊 Output Format

Generated parsers return pandas DataFrames with the following schema:

| Column | Type | Description |
|--------|------|-------------|
| Date | string | Transaction date (DD-MM-YYYY) |
| Description | string | Transaction description |
| Debit Amt | float | Debit amount (pd.NA if credit) |
| Credit Amt | float | Credit amount (pd.NA if debit) |
| Balance | float | Account balance after transaction |

## 🎯 Key Components

### Pydantic Models

- **PDFAnalysis**: Structured analysis of PDF content and parsing strategy
- **ParserCode**: Generated parser code with imports and explanation
- **TestResult**: Test execution results and validation metrics
- **FixPlan**: Issue identification and fix strategies

### Agent State

The agent maintains state throughout execution:
```python
class AgentState(TypedDict):
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
```

## 📈 Success Metrics

The agent validates success through:

- **Schema Alignment**: Correct column names and data types
- **Row Count Accuracy**: Matches expected transaction count
- **Data Quality**: Valid dates, non-empty descriptions, proper amounts
- **Sample Validation**: First 3 rows match expected values exactly
- **Balance Logic**: Debit/credit classification accuracy

## 🔧 Customization

### Adding New Banks

1. Create data folder: `data/{bank_name}/`
2. Add sample PDF: `{bank_name} sample.pdf`
3. Add expected results: `result.csv`
4. Run: `python agent.py --target {bank_name}`

### Modifying Analysis Depth

Adjust the analysis prompt in `analyze_pdf_node()` to focus on specific aspects:
- Pattern recognition depth
- Edge case identification
- Regex complexity
- Error handling requirements

### Changing LLM Provider

Modify `start.py` to use different LLM providers:
```python
# Update client creation and model names
client = create_openai_client(api_key, base_url)
```



### Analysis Output Example

```
Step 1: Performing comprehensive PDF analysis...
Strategy: Multi-page regex-based extraction with balance change logic
Edge cases identified: 5
Regex patterns recommended: 3

Step 2: Generating parser code from comprehensive analysis...
Parser code generated: custom_parsers/icici_parser.py

Step 3: Testing parser...
Parser test passed!
```

## 🐛 Troubleshooting

### Common Issues

1. **PDF Reading Errors**
   - Install both PyPDF2 and pdfplumber
   - Check PDF file permissions and format

2. **LLM Connection Issues**
   - Verify API keys in `.env` file
   - Check endpoint URL and model availability

3. **Parsing Failures**
   - Review PDF text extraction quality
   - Adjust regex patterns for specific formats
   - Check multi-page handling logic

### Debug Mode

Enable detailed logging by modifying the agent state or adding debug prints in node functions.
