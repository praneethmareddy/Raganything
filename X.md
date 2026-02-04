Perfect 👍 — this simplifies everything.
Let’s do a pure, basic Drools “Hello-World style transformation” with ZERO CIQ, ZERO Excel, ZERO complexity.

This is the smallest possible working Drools project to prove your setup is correct.


---

🎯 WHAT THIS SIMPLE PROJECT DOES

Input: one simple fact

Rule fires

Output: two objects created

No CIQ, no XLSX, no conditions


This is only to verify:

> “Drools rule compilation + execution works”




---

📦 PROJECT ASSUMPTION (important)

Business Central created your project under:

com.myspace.simple_demo

👉 We will use this package everywhere
👉 Do NOT change it


---

🧹 STEP 0 — CLEAN PROJECT (MANDATORY)

Inside Business Central → simple_demo project:

Delete everything:

All Data Objects

All Rules


You should see NO assets.


---

STEP 1 — Create ONE SIMPLE INPUT OBJECT

Step name: InputFact

1. Add Asset → Data Object


2. Fill:

Name: InputFact

Package: com.myspace.simple_demo



3. Click OK



Add fields:

Field	Type

value	String


Click Save

✅ This is the only input


---

STEP 2 — Create ONE OUTPUT OBJECT

Step name: OutputFact

1. Add Asset → Data Object


2. Fill:

Name: OutputFact

Package: com.myspace.simple_demo



3. Click OK



Add fields:

Field	Type

message	String


Click Save


---

STEP 3 — Create BASIC DRL RULE

Step name: basic-transform

1. Add Asset → Rule (DRL Rule)


2. Fill:

Name: basic-transform

Package: com.myspace.simple_demo



3. Click OK




---

Paste THIS EXACT RULE (VERY IMPORTANT)

package com.myspace.simple_demo

rule "Basic Transformation Rule"
when
    $in : InputFact()
then
    OutputFact out = new OutputFact();
    out.setMessage("Hello Drools, input = " + $in.getValue());

    insert(out);
end

Click Save

✅ No imports
✅ No conditions
✅ No CIQ
✅ No Excel


---

STEP 4 — BUILD PROJECT

1. Click Build



✅ SUCCESS MUST BE:

Build successful

If this fails → tell me the error text.


---

STEP 5 — DEPLOY PROJECT

1. Click Deploy




---

STEP 6 — DEPLOY TO KIE SERVER

Menu ☰ → Deploy → Execution Servers

Add container:

Field	Value

Container ID	simple-demo
Alias	simple-demo
KIE Base	default
KIE Session	default


Status must be:

STARTED


---

STEP 7 — TEST THE RULE (ONE COMMAND)

Run this from terminal:

curl -u admin:admin \
  -H "Content-Type: application/json" \
  -X POST \
  http://<SERVER_IP>:8180/kie-server/services/rest/server/containers/simple-demo \
  -d '{
    "commands": [
      {
        "insert": {
          "object": {
            "com.myspace.simple_demo.InputFact": {
              "value": "TEST"
            }
          }
        }
      },
      {
        "fire-all-rules": {}
      }
    ]
  }'


---

✅ SUCCESS OUTPUT (LOGICAL)

Response JSON will contain:

"message": "Hello Drools, input = TEST"

That proves:

✔ DRL compiled
✔ Rule executed
✔ Object created
✔ KIE Server works
✔ Docker setup is correct


---

🧠 WHY THIS IS THE RIGHT BASELINE

No CIQ confusion

No Excel parsing

No package mismatch

No “cannot resolve type”

Smallest Drools project possible


This is the baseline every Drools setup should pass.


---

🔜 NEXT STEPS (ONLY AFTER THIS WORKS)

Choose ONE:

1️⃣ Add second output object
2️⃣ Rename output to @acpf / @serverinfo logic
3️⃣ Re-introduce CIQ step-by-step
4️⃣ Move this exact project to Kubernetes

Reply with 1 / 2 / 3 / 4 and we’ll continue cleanly 👍
