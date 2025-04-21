# Data Lineage Tracker Playbook

## Overview
The Data Lineage Tracker systematically identifies and tracks the origin and transformations of data elements within diverse applications, including mainframe and legacy systems, to support governance, regulatory compliance, and impact analysis. This playbook guides Application Owners and Technical SMEs through preparing data inputs, identifying key scripts and mainframe elements, and validating lineage outcomes.

## Roles and Responsibilities
- **Application Owners & Technical SMEs:** Provide and prepare input data and application codebases, filter irrelevant data, document critical scripts and mainframe processes, and validate lineage results.
- **Data Lineage Tracker Team:** Processes inputs, tracks lineage transformations, and generates lineage outputs.

## Step-by-Step Process

### Step 1: Preparing Input Data

#### 1.1 Folder Structure
Organize application metadata, scripts, and mainframe elements clearly and consistently in folders:

```
Application_Folder
├── Database_1
│   ├── Schema_A
│   │   ├── Table_X
│   │   ├── Table_Y
│   │   └── Supporting_Scripts (optional)
│   └── Schema_B
│       ├── Table_Z
│       └── Supporting_Scripts (optional)
├── Database_2
│   └── Schema_C
│       ├── Table_W
│       └── Supporting_Scripts (optional)
├── Mainframe (if applicable)
│   ├── JCL_Jobs
│   ├── COBOL_Programs
│   └── Other_Scripts_and_Documentation
└── Supporting_Scripts (optional, at Application or Database level)
```

- Clearly list columns within each `Table_X` folder.
- Include supporting scripts containing additional logic (driver scripts, helper scripts, mainframe scripts, and programs).

#### 1.2 Filtering Irrelevant Information
- SMEs proactively filter out historical tables, backup databases, redundant mainframe jobs, or scripts.
- Document explicitly any data or scripts intentionally excluded.

### Step 2: Providing Application Codebase

#### 2.1 Codebase and Mainframe Organization
- Document and package the production codebase and mainframe code clearly.
- Support all programming languages, scripts, driver scripts, JCL, COBOL programs, and other mainframe-specific components.
- Clearly label main scripts, mainframe entry points, and job sequences.

#### 2.2 Supporting Documentation
- Provide documentation explaining the main logic, scripts, and mainframe processes used for data transformations.
- Include clear comments in scripts and code to facilitate lineage analysis.

### Step 3: Processing and Tracking Lineage

- Data Lineage Tracker analyzes provided scripts, supporting scripts, mainframe programs, and metadata to track upstream lineage, beginning from downstream data elements and tracing them back to their origins.

### Step 4: Reviewing and Validating Lineage Output

#### 4.1 Lineage Output Format
- Results are delivered in Excel files outlining data lineage relationships clearly.
- Excel files include four-part keys (`database | schema | table | column`) and their associated lineage tracking information.

**Example Lineage Excel:**

| Database | Schema | Table  | Column      | Parent Data Element (tracked by DLT)         |
|----------|--------|--------|-------------|---------------------------------------------|
| Sales_DB | Fin    | Sales  | Total_Sales | Sales_DB.Fin.Transactions.Sales_Amount |

#### 4.2 SME Validation
- SMEs validate lineage accuracy by carefully reviewing Excel outputs.
- Flag ambiguities or incomplete lineage entries explicitly for further manual review or deeper analysis by the Data Lineage Tracker team.

### Edge Case Handling
- Explicitly flag ambiguous or incomplete lineage entries.
- Mark flagged entries for manual or advanced review by the Data Lineage Tracker team.

## Best Practices & Recommendations
- Clearly document and communicate the structure and specifics of your codebase and mainframe components.
- Ensure initial metadata provided is accurate, thoroughly filtered, and representative of the application.
- Regularly coordinate with the Data Lineage Tracker team for continuous improvement and to address specific technical or domain-related challenges.

## Questions & Support
For support or further assistance, please contact the Data Lineage Tracker implementation team.

