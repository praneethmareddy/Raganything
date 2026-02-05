Perfect. I’ll now explain exactly what you do inside Drools Workbench, screen by screen, and why each thing exists.
I’ll also clearly separate INPUT → RULE → OUTPUT, so you understand the mental model, not just syntax.

I’ll assume:

You already have Drools Workbench + KIE Server running in Docker

You want CIQ (XLSX rows) → 2 sections (@acpf, @serverinfo)



---

0️⃣ First fix one thing (important)

You wrote “dslr” — the correct file is DRL
👉 Drools Rule Language (.drl)


---

1️⃣ What Drools Workbench is doing (big picture)

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
