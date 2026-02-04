Perfect 👍
We’ll do this exactly in the right order:

1️⃣ Docker first (mandatory baseline)
2️⃣ Then Kubernetes (same setup, production-style)

I will assume your server is EMPTY and explain what to do, why you do it, and what “success” looks like at each step.


---

PHASE 1 — DOCKER (START FROM ZERO)

Goal of this phase

Get Drools Workbench and KIE Server running

Validate URLs

Understand how the two components exist as containers

No Kubernetes yet



---

1️⃣ Understand what you are deploying (once)

You will run two Docker containers from Drools:

Container	Purpose	Port

Drools Workbench (Business Central)	Rule authoring UI	8080
KIE Server	Rule execution (REST API)	8180


Flow (important):

Rule Author → Workbench
CIQ App     → KIE Server


---

2️⃣ Install Docker on empty server

2.1 Login to server

ssh user@server_ip

2.2 Install Docker

RHEL / CentOS / Rocky

sudo yum install -y docker

Ubuntu

sudo apt update
sudo apt install -y docker.io

2.3 Start Docker

sudo systemctl start docker
sudo systemctl enable docker

2.4 Verify

docker --version
docker ps

✅ If no error → Docker is ready


---

3️⃣ Run Drools Workbench (Docker)

This is the UI.

3.1 Start container

docker run -d \
  --name drools-workbench \
  -p 8080:8080 \
  jboss/drools-workbench-showcase

3.2 Verify container

docker ps

You must see:

drools-workbench   Up

3.3 Access UI

Open browser:

http://<SERVER_IP>:8080/business-central

Login:

username: admin
password: admin

✅ If UI opens → Workbench is OK


---

4️⃣ Run KIE Server (Docker)

This is the rule engine.

4.1 Start container

docker run -d \
  --name kie-server \
  -p 8180:8080 \
  jboss/kie-server-showcase

4.2 Verify

docker ps

4.3 Test endpoint

Open:

http://<SERVER_IP>:8180/kie-server/services/rest/server

Expected:

JSON / auth prompt / HTTP response


✅ If it responds → KIE Server is running


---

5️⃣ Link Workbench ↔ KIE Server (Docker only)

This allows:

Workbench to deploy rules

KIE Server to execute rules


docker stop kie-server
docker rm kie-server

docker run -d \
  --name kie-server \
  --link drools-workbench:kie_wb \
  -p 8180:8080 \
  jboss/kie-server-showcase


---

6️⃣ Docker success checklist (VERY IMPORTANT)

Run:

docker ps
docker logs drools-workbench
docker logs kie-server

You must confirm:

Both containers are Up

UI reachable on 8080

KIE Server reachable on 8180


👉 STOP HERE if this doesn’t work
👉 Kubernetes comes ONLY after this is stable


---

PHASE 2 — KUBERNETES (AFTER DOCKER WORKS)

Now we move the same containers into Kubernetes.


---

7️⃣ Install Kubernetes (single-node, easiest)

Since this is one server, use Minikube.

7.1 Install kubectl

curl -LO https://storage.googleapis.com/kubernetes-release/release/`curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt`/bin/linux/amd64/kubectl
chmod +x kubectl
sudo mv kubectl /usr/local/bin/

Verify:

kubectl version --client


---

7.2 Install Minikube

curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
chmod +x minikube-linux-amd64
sudo mv minikube-linux-amd64 /usr/local/bin/minikube


---

7.3 Start Kubernetes

minikube start --driver=docker

Verify:

kubectl get nodes

You must see:

Ready


---

8️⃣ Create namespace

kubectl create namespace drools


---

9️⃣ Deploy Drools Workbench (K8s)

9.1 Deployment

apiVersion: apps/v1
kind: Deployment
metadata:
  name: drools-workbench
  namespace: drools
spec:
  replicas: 1
  selector:
    matchLabels:
      app: drools-workbench
  template:
    metadata:
      labels:
        app: drools-workbench
    spec:
      containers:
      - name: drools-workbench
        image: jboss/drools-workbench-showcase
        ports:
        - containerPort: 8080

Apply:

kubectl apply -f drools-workbench-deployment.yaml


---

9.2 Service

apiVersion: v1
kind: Service
metadata:
  name: drools-workbench
  namespace: drools
spec:
  type: NodePort
  selector:
    app: drools-workbench
  ports:
  - port: 8080
    targetPort: 8080
    nodePort: 30080

Apply:

kubectl apply -f drools-workbench-service.yaml

Access:

http://<MINIKUBE_IP>:30080/business-central


---

🔟 Deploy KIE Server (K8s)

10.1 Deployment

apiVersion: apps/v1
kind: Deployment
metadata:
  name: kie-server
  namespace: drools
spec:
  replicas: 2
  selector:
    matchLabels:
      app: kie-server
  template:
    metadata:
      labels:
        app: kie-server
    spec:
      containers:
      - name: kie-server
        image: jboss/kie-server-showcase
        ports:
        - containerPort: 8080

Apply:

kubectl apply -f kie-server-deployment.yaml


---

10.2 Service

apiVersion: v1
kind: Service
metadata:
  name: kie-server
  namespace: drools
spec:
  type: NodePort
  selector:
    app: kie-server
  ports:
  - port: 8080
    targetPort: 8080
    nodePort: 30180

Apply:

kubectl apply -f kie-server-service.yaml

Access:

http://<MINIKUBE_IP>:30180/kie-server/services/rest/server


---

11️⃣ Final verification

kubectl get pods -n drools
kubectl get svc -n drools

You should see:

1 workbench pod

Multiple kie-server pods

Services exposed



---

12️⃣ What you should clearly understand now

Docker = baseline validation

Kubernetes = scaling + production

Workbench = rule authoring

KIE Server = rule execution

CIQ parsing & config generation are outside Drools



---

Next step (tell me)

I can now: 1️⃣ Show exact CIQ row → REST call → rule output
2️⃣ Explain how rules created in Workbench reach KIE Server
3️⃣ Show end-to-end CIQ → config lifecycle diagram
4️⃣ Help you prepare architecture/design explanation for interview/review

Just tell me what you want next.
