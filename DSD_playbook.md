# Data Source Dictionary (DSD) Playbook

## Overview
The Data Source Dictionary (DSD) standardizes and enriches business metadata (business names and descriptions) across various applications to enhance data governance, improve consistency, and provide clarity to stakeholders. This playbook serves as a comprehensive guide for metadata stewards, detailing the steps required to prepare, review, and finalize metadata using the DSD methodology.

## Roles and Responsibilities
- **Metadata Stewards:** Responsible for preparing initial metadata, reviewing AI-generated predictions, making necessary edits, and ensuring metadata quality.
- **DSD Team:** Responsible for identifying ambiguous data elements, generating AI-assisted predictions, and facilitating the overall metadata integration process.

## Step-by-Step Process

### Step 1: Preparing Input Data

#### 1.1 Collecting Representative Samples
- Collect a representative set of existing metadata (technical names matched with corresponding business names and descriptions).
- This sample ensures AI predictions are contextually accurate and relevant.

#### 1.2 Handling Ambiguous Data
- DSD Team identifies potentially challenging or ambiguous metadata entries.
- Metadata stewards manually label these entries clearly and comprehensively to ensure accurate AI-generated recommendations.

#### 1.3 Data Submission Format
- All metadata, including labeled and manually reviewed samples, should be submitted using an Excel file.

**Example Excel Format:**

| Technical Name | Business Name       | Description                        |
|----------------|---------------------|------------------------------------|
| `ACCT_BAL`     | Account Balance     | The balance currently in the account.|
| `CUST_ID`      | Customer Identifier | Unique identifier assigned to a customer.|

---

### Step 2: Reviewing and Validating Predictions

#### 2.1 AI-Assisted Recommendations
- The DSD Team generates the top-3 AI-driven predictions for business names and descriptions based on provided samples.
- Predictions are presented clearly in an Excel file for easy review.

#### 2.2 SME Review Process
- Metadata stewards (SMEs) review the predictions provided by the DSD.
- SMEs select the most suitable prediction or edit/rewrite the metadata as necessary.

**Example of SME Review:**

| Technical Name | Suggested Business Names (Top 3)                           | Finalized Business Name |
|----------------|-------------------------------------------------------------|-------------------------|
| `CRDT_LMT_AMT` | Credit Limit Amount; Credit Limit; Customer Credit Threshold| **Credit Limit**        |
| `TXN_DT`       | Transaction Date; Date of Transaction; Transaction Timestamp | **Transaction Date**    |

---

### Step 3: Finalization and Submission

#### 3.1 Final Accuracy Checks
- SMEs perform thorough accuracy checks on selected or edited metadata to ensure high quality.
- Ensure consistency in naming conventions and clarity of descriptions.

#### 3.2 Submitting Reviewed Metadata
- Once validated, the reviewed and finalized Excel file is returned to the DSD Team.
- Clearly mark the file as finalized and approved for integration.

#### 3.3 Integration into Downstream Applications
- The finalized and approved business metadata is integrated by the DSD Team into downstream applications.
- This integration ensures metadata consistency and improved governance across the organization.

---

## Best Practices & Recommendations
- Prioritize clarity, accuracy, and consistency when selecting and reviewing metadata labels.
- Clearly identify and proactively manually label ambiguous terms early in the process to enhance AI prediction accuracy.
- Maintain ongoing communication with the DSD Team to continuously highlight and address domain-specific nuances or challenges.
- Establish regular checkpoints and feedback loops between metadata stewards and the DSD Team to ensure smooth and continuous improvements.

---

## Support & Additional Resources
- For support, assistance, or clarification at any stage, please contact the DSD implementation team.
- Additional resources, training materials, and guidelines may be provided upon request.

