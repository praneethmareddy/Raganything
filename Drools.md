Below is a clean, neat, GitHub-ready Markdown guide that explains exactly what to do inside Drools Workbench, screen by screen, with a clear INPUT → RULE → OUTPUT mental model.

You can copy-paste this as documentation or follow it step by step.


---

CIQ (XLSX) → NE Template Mapping using Drools Workbench

Assumptions

Drools Workbench and KIE Server are already running in Docker

Workbench is connected to KIE Server

Input is CIQ Excel (XLSX)

Output must contain 2 sections per row:

@acpf

@serverinfo




---

0️⃣ Important Clarification

The correct rule file is DRL

DRL = Drools Rule Language

There is no “dslr” in Drools



---

1️⃣ What Drools Workbench Does (Big Picture)

Workbench Responsibilities

Define Input model (facts)

Define Output model

Define Rules (mapping logic)

Package everything into a KJAR

Deploy rules to KIE Server


Workbench Does NOT Do

Read Excel files

Loop through rows

Write output files


> Excel handling and file writing happen outside Drools




---

2️⃣ Create a Project (KJAR)

Step 2.1 — Login

http://<ip>:8080/business-central

Login with:

kieadmin / kieadmin

Step 2.2 — Create Project

Navigate to:

Menu → Design → Projects → Add Project

Fill details:

Name        : ciq-to-ne
Group ID    : com.telco.rules
Artifact ID : ciq-ne-mapping
Version     : 1.0.0

✅ This project is your rule container (KJAR)


---

3️⃣ Define INPUT (One Excel Row)

Concept

> Each Excel row = ONE input fact




---

Step 3.1 — Create Input Data Object

Path:

Project → Assets → Data Objects → Add Data Object

Fill:

Name    : CIQRow
Package : com.telco.ciq

Add Fields

Field Name	Type	Source

neId	Integer	Excel
neName	String	Excel
neType	String	Excel


✅ Represents one row of CIQ XLSX
✅ Getters/setters are auto-generated


---

4️⃣ Define OUTPUT (Generated Config Sections)

Concept

> One CIQRow → multiple output sections




---

Step 4.1 — Create Output Data Object

Path:

Assets → Data Objects → Add Data Object

Fill:

Name    : OutputBlock
Package : com.telco.output

Add Fields

Field	Type	Meaning

section	String	acpf / serverinfo
text	String	config content


✅ One OutputBlock = one config section


---

5️⃣ Write Rules (Mapping Logic)

Step 5.1 — Create DRL File

Path:

Assets → Add Asset → DRL File

Name:

ciq-to-template


---

Step 5.2 — DRL Header

package com.telco.rules

import com.telco.ciq.CIQRow
import com.telco.output.OutputBlock

Why

CIQRow → input

OutputBlock → output



---

Step 5.3 — Rule for @acpf Section

rule "CIQ row to ACPF section"
when
    $row : CIQRow()
then
    String cfg =
        "@acpf\n" +
        "NE ID  NE Name  NE Type\n" +
        $row.getNeId() + "  " +
        $row.getNeName() + "  " +
        $row.getNeType() + "\n";

    OutputBlock out = new OutputBlock();
    out.setSection("acpf");
    out.setText(cfg);

    insert(out);
end

What this rule does

Matches one CIQ row

Builds the @acpf section

Inserts output into working memory



---

Step 5.4 — Rule for @serverinfo Section

rule "CIQ row to ServerInfo section"
when
    $row : CIQRow()
then
    String cfg =
        "@serverinfo\n" +
        "NE ID  NE Name\n" +
        $row.getNeId() + "  " +
        $row.getNeName() + "\n";

    OutputBlock out = new OutputBlock();
    out.setSection("serverinfo");
    out.setText(cfg);

    insert(out);
end

Key Point

Same input fact

Different mapping

Different output section



---

6️⃣ Internal Execution Model (CRITICAL)

For each CIQRow inserted:

Rule 1 → OutputBlock(acpf)
Rule 2 → OutputBlock(serverinfo)

So:

1 CIQ row  →  2 output blocks
N CIQ rows →  2 × N output blocks

✅ Exactly matches your requirement


---

7️⃣ Deploy Rules

Click Deploy (top-right in Workbench)

What happens automatically:

DRL compiled

KJAR created

Deployed to KIE Server

Rules become executable



---

8️⃣ How Excel Data Reaches Drools

> Drools never reads XLSX directly



External application flow:

Excel
 ↓
Read rows
 ↓
Convert row → CIQRow JSON
 ↓
Send to KIE Server

Example REST Payload (one row)

{
  "commands": [
    {
      "insert": {
        "object": {
          "com.telco.ciq.CIQRow": {
            "neId": 1,
            "neName": "Acpf",
            "neType": "Acpf"
          }
        }
      }
    },
    { "fire-all-rules": {} }
  ]
}


---

9️⃣ Output You Collect

Returned OutputBlock objects:

@acpf
NE ID  NE Name  NE Type
1      Acpf     Acpf

@serverinfo
NE ID  NE Name
1      Acpf

Your application:

Appends blocks

Writes final .cfg file



---

🔟 Why This Design Is Correct

Principle	Reason

One row = one fact	Clean & scalable
One section = one rule	Simple mapping
No loops in DRL	Best practice
Excel handled outside	Stability
Rules are dynamic	Business-friendly



---

11️⃣ Common Mistakes (Avoid)

❌ Reading Excel inside Drools

❌ One giant rule for full file

❌ Building full config in a single rule

❌ Using globals for everything



---

12️⃣ Next Steps (Optional)

Choose one to go deeper:

1. Excel Decision Table instead of DRL


2. Full Python + Drools REST example


3. Conditional sections


4. Multi-NE-type templates


5. Validation + transformation together



Just tell me the number.
Workbench responsibility

Define input model

Define output model

Define rules (mapping logic)

Package everything as a KJAR


NOT Workbench’s job

Reading Excel

Looping rows

Writing files


Those happen outside.


---

2️⃣ Step-by-step in Drools Workbench (NO SKIPS)


---

STEP 2.1 — Create a Project (KJAR)

1. Login to Workbench

http://<ip>:8080/business-central


2. Go to
Menu → Design → Projects


3. Click Add Project



Fill like this:

Name        : ciq-to-ne
Group ID    : com.telco.rules
Artifact ID : ciq-ne-mapping
Version     : 1.0.0

👉 This project is your rule container


---

3️⃣ Define INPUT (this represents ONE Excel row)

STEP 3.1 — Create CIQRow (Input Fact)

Path:

Project → Assets → Data Objects → Add Data Object

Fill:

Name    : CIQRow
Package : com.telco.ciq

Add fields (VERY IMPORTANT)

Field Name	Type	Why

neId	Integer	From Excel
neName	String	From Excel
neType	String	From Excel


✔ This object = one row of XLSX
✔ Workbench auto-generates getters/setters


---

Mental model (remember this)

> Every Excel row = one CIQRow fact




---

4️⃣ Define OUTPUT (what rules will generate)

STEP 4.1 — Create OutputBlock

Path:

Assets → Data Objects → Add Data Object

Fill:

Name    : OutputBlock
Package : com.telco.output

Fields

Field	Type	Why

section	String	acpf / serverinfo
text	String	actual config text


✔ One OutputBlock = one section of output


---

Mental model

> 1 CIQRow → multiple OutputBlock objects




---

5️⃣ Write RULES (this is the mapping logic)

Path:

Assets → Add Asset → DRL File

Name:

ciq-to-template


---

STEP 5.1 — DRL header (MANDATORY)

package com.telco.rules

import com.telco.ciq.CIQRow
import com.telco.output.OutputBlock

Why

CIQRow → input

OutputBlock → output



---

STEP 5.2 — Rule for @acpf section

rule "CIQ row to ACPF section"
when
    $row : CIQRow()
then
    String cfg =
        "@acpf\n" +
        "NE ID  NE Name  NE Type\n" +
        $row.getNeId() + "  " +
        $row.getNeName() + "  " +
        $row.getNeType() + "\n";

    OutputBlock out = new OutputBlock();
    out.setSection("acpf");
    out.setText(cfg);

    insert(out);
end

What this rule is doing (line by line)

Line	Meaning

CIQRow()	Match ONE Excel row
@acpf	Section header
Uses 3 fields	Mapping logic
insert(out)	Send output back


✔ Fires once per CIQ row


---

STEP 5.3 — Rule for @serverinfo section

rule "CIQ row to ServerInfo section"
when
    $row : CIQRow()
then
    String cfg =
        "@serverinfo\n" +
        "NE ID  NE Name\n" +
        $row.getNeId() + "  " +
        $row.getNeName() + "\n";

    OutputBlock out = new OutputBlock();
    out.setSection("serverinfo");
    out.setText(cfg);

    insert(out);
end

Key point

Same input (CIQRow)
Different output mapping
Different section

✔ That’s how 2 sections are generated


---

6️⃣ What happens internally (CRITICAL UNDERSTANDING)

For each CIQRow inserted:

Rule 1 fires → OutputBlock(acpf)
Rule 2 fires → OutputBlock(serverinfo)

So:

1 row → 2 output blocks
N rows → 2 × N output blocks

Exactly your requirement.


---

7️⃣ Deploy (make rules active)

Click Deploy (top-right)

What happens:

Rules compiled

KJAR created

Automatically deployed to KIE Server



---

8️⃣ How INPUT reaches Drools (Excel part)

Drools never sees XLSX directly.

Your application does:

Excel
 ↓
Read rows
 ↓
Convert each row → CIQRow JSON
 ↓
Send to KIE Server

Example input JSON (row 1):

{
  "commands": [
    {
      "insert": {
        "object": {
          "com.telco.ciq.CIQRow": {
            "neId": 1,
            "neName": "Acpf",
            "neType": "Acpf"
          }
        }
      }
    },
    { "fire-all-rules": {} }
  ]
}


---

9️⃣ What OUTPUT you collect

Drools returns OutputBlock objects:

@acpf
NE ID  NE Name  NE Type
1      Acpf     Acpf

@serverinfo
NE ID  NE Name
1      Acpf

Your app:

Appends them

Writes to .cfg file



---

🔟 Why this is the CORRECT design

Reason	Why

One row = one fact	Clean logic
One section = one rule	Easy mapping
No loops in DRL	Drools best practice
XLSX handled outside	Stable
Rules fully dynamic	Business-friendly



---

11️⃣ Common mistakes (avoid these)

❌ Trying to read Excel inside Drools
❌ Writing one giant rule for whole file
❌ Concatenating full file inside DRL
❌ Using globals for everything


---

12️⃣ If you want next (pick one)

1️⃣ Same logic using Excel Decision Table (no DRL coding)
2️⃣ Full Python script + Drools REST integration
3️⃣ Add conditional sections
4️⃣ Multi-NE-type templates
5️⃣ Validation + transformation together

Say the number — I’ll go equally deep.
