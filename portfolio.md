# **BisiPortfolio**

## 🛠️ Philosophy & Background

### Dive down to the atom. Execute without rest.
I am an AI Systems Engineer who doesn't rely on quick talent; I rely on obsessive perseverance and strict technical integrity. I build following a first-principles approach: when a complex system catches my attention, I dive deep into its core mechanics, from optimizing asynchronous cloud architectures to squeezing inference latency on edge hardware.

My core focus is bridging the gap between advanced deep learning research (Self-Supervised Learning, Vision Transformers) and real-world Physical AI. This means implementing rigorous LLMOps to optimize pipelines for minimal latency, high throughput, and stable automation under strict computational and industrial hardware constraints.

---

### 📈 The Journey: Building a Self-Taught Engine

My academic and professional trajectory has been shaped by resilience, rapid adaptation, and extreme self-discipline.

* **The Turnaround:** I started my Computer Engineering BSc at Politecnico di Torino during the 2020 pandemic. Disoriented by remote learning and isolation, my first year was a failure—leaving me at the bottom of my peer group with only two passed exams.Refusing to accept this outcome, I built an intensive, 100% self-taught methodology from my bedroom.
* **The Acceleration:** Once this internal engine clicked, my execution speed exploded. In the summer of 2023, I shattered the academic backlog by passing **7 hardcore engineering exams in just 3 months** (including core pillars like *Operating Systems*, *Computer Architecture*, and *Algorithms & Data Structures*). This momentum allowed me to overtake my peer group, graduate early, and jump straight into an AI & Data Analytics MEng by September 2023.
* **Strategic Anticipation:** I cleared my first Master’s level courses (*Big Data Processing* and *Data Science*) in early February 2024, capitalizing on the winter exam session **before even formally receiving my Bachelor’s degree** in March 2024.

---

### 🏭 From Startup to Industry: Compartmentalization & Core Metrics

To complete my BSc, I chose to invest extra time into a hands-on internship (Nov 2023 – Feb 2024) as a **Junior AI Engineer at Sentric**, a fast-scaled startup managing millions of data points for agentic commerce pipelines. This experience shifted my career vector. 

In May 2024, I was brought into **Simplex Rapid**, an international heavy-industry manufacturer of precision CNC machinery, as an **ML Engineer** to architect their AI infrastructure from scratch, deploying production-grade multimodal and vision-language automation. 

During this period, I pushed my capacity for extreme time compartmentalization to its absolute limit:
* **September 2024:** While actively managing production-level AI pipelines under industrial constraints, I isolated my focus to study and clear **4 core Master's exams in exactly 14 days** without delaying corporate shipping timelines.
* **The Promotion:** Within a year, the systems I deployed successfully **reduced manual manufacturing effort by 71%**. Based on these results, in May 2025, I was promoted to **Technical Lead** and offered a permanent management relocation to Milan to scale AI across their entire production line.

**I walked away from the offer.** I don't look for safe harbors or comfortable positions; I look for massive technical challenges. I allowed the contract to expire naturally to transition into world-class deep-tech ecosystems between Zurich and Silicon Valley. I sealed this chapter by scoring **30/30 cum laude in Machine Learning in Applications**, turning my final research project on Self-Supervised Learning into an ongoing international publication effort with pathology research centers in Paris, Lyon, and Nice.

I move fast, learn down to the bare metal, and view high-pressure environments as the ultimate catalyst for innovation.

---

## 💼 Professional Experience

### **ML Engineer (Technical Lead) — Simplex Rapid** *(05/2024 - 10/2025)*
Architected and deployed end-to-end LLM and vision-language systems for a global leader in precision CNC machinery, operating under strict industrial constraints. 
* **Multilingual Pipeline:** Designed a fine-tuned translation pipeline across **20 languages**, achieving a **~53% touchless rate**. Reduced total human effort from **~8,000 to ~2,350 hours (71% reduction)** and cut linguistic processing costs by **~76%**.
* **Vision-to-Manufacturing:** Built an LLM+vision extraction flow that reads complex spring engineering drawings/PDFs with per-field validation and a human-in-the-loop review, automating the population of manufacturing parameters.
* **Agentic Ops:** Migrated an internal business assistant to the Responses API with tool calling, implementing comprehensive monitoring and tracing for reliability and auditability.
* **Stack:** Python, PyTorch, OpenAI APIs, VLM, Agentic Workflows, MLOps.

### **Junior AI Engineer (Intern) — Sentric** *(11/2023 - 02/2024)*
* Focused on applied R&D for large-scale e-commerce datasets. Engineered backend data pipelines and automated complex data cleaning processes using LLM integrations. The high execution speed and problem-solving demonstrated here led directly to my recruitment as ML Engineer at Simplex Rapid.

---

## 🔬 Applied Research

### **SSL for RCC Classification** *(2026)*
* **Focus:** Self-Supervised Learning (SSL), Computational Pathology, Vision Transformers.
* **Status:** Preliminary phase completed with **30/30 cum laude**. Ongoing international collaboration with pathology centers (Paris, Lyon, Nice) to transition the research into a formal publication. *(Code & dataset private pending publication).*

### **[Humanoid Motion Diffusion](https://github.com/Blackhand01/Humanoid-Motion-Diffusion)** *(2026)*
* **Focus:** Generative robotics, multimodal motion synthesis, Sim-to-Real trajectory validation.
* **Architecture:** Audio-conditioned whole-body humanoid trajectory synthesis using DDPM diffusion transformers over 24-joint SMPL pose sequences (AIST++). 
* **Metrics:** Dropped Temporal Smoothness Index from `12.60` to `0.08` utilizing EMA/CFG on dense 120-frame clips. Optimized deployment metrics include JLVR `6.0%`, BAS `0.19`, and self-collision risk at `0.0004`.

### **[Affordance 3D Highlighting](https://github.com/Blackhand01/Affordance_Highlighting_Project_2024)** *(2025)*
* **Focus:** Embodied AI, zero-shot 3D localization.
* Bridged natural language reasoning with 3D spatial environments to zero-shot localize functional regions from language prompts, inspired by 3D Highlighter (CVPR 2023).

---
## Edge Systems, Robotics & Hackathons

### **[Autonomous Swarm - Mirai / Fincantieri Hackathon](https://github.com/MatiasSalaris/maritime_task_4/tree/demo)** *(2026)*

Engineered a fault-tolerant LLM-driven multi-agent system for USV/UAV maritime autonomy.

* Implemented dynamic failure recovery and shared situational awareness for edge environments.
* Designed agent coordination under partial information, mission constraints, and runtime failures.
* Relevant to robust autonomy, multi-agent systems, and real-time decision loops.

---

### **[Edge-VLA-Micro: Distributed VLA Stack for Edge Autonomy](https://github.com/Blackhand01/Edge-VLA-Micro)** *(2026)*
A low-latency Voice-to-Action robotics stack for PX4 drones. The system runs in two profiles: a Mac-only research profile with Qwen2-VL over MLX, and a distributed edge profile where the Mac acts as a smart sensor node (ASR + camera) while a Jetson Orin Nano runs the VLA/control brain with SmolVLM on CUDA.
* **Safety & Control:** The VLM is not a control authority. Perception proposes intent, deterministic Pydantic/CV/state-machine guards authorize it, and MAVSDK executes only validated PX4 commands.
* **Edge Deployment:** Demonstrated on Jetson Orin Nano 8GB with SmolVLM-256M, FastAPI bridge, OpenCV red-target guardrails, MAVSDK/PX4 integration, and hardware telemetry logging.
* **Measured Stability:** Clean distributed run peaked at **3.5GB / 7.6GB RAM**, **99% GPU load during visual inference**, and **50.66 C max temperature**.
* **Bottleneck Analysis:** The hardware was stable; the dominant latency was software-side ASR and visual VLM inference. Measured distributed profile: Mac ASR ~**11.16s**, Jetson SmolVLM inference ~**4.04s** average, visual inference throughput ~**10.5 TPS**.

### **[Embedded Vision Trade-offs (Arduino Nicla Vision)](https://github.com/Blackhand01/embedded-vision-tradeoffs-m7)** *(2026)*
Exposed the critical trade-offs between accuracy, latency, RAM, and Flash through hardware-in-the-loop benchmarking. Demonstrates why accuracy-only comparisons fail on bare-metal targets by systematically surfacing activation peaks and memory bottlenecks.

### **[Multi-Agent Advisor System (ETH AI Center / UBS Track)](https://github.com/Blackhand01/ZurichHackathon2025)** *(2025)*
Co-built a multi-agent dialog-to-action prototype using Apertus-8B and GPT-4o. Features a lightweight intent classifier supervised by a "judge" model that arbitrates disagreements and triggers deterministic safe human handoffs. 

### **[MuseINO (Tiny Hack Torino / Italian Tech Week)](https://github.com/Blackhand01/TinyHack2025)** *(2025)*
Developed an edge-based attention-tracking device using Arduino Nicla Vision to quantify dwell time and engagement without recording video, ensuring complete privacy. MVP built and iterated in 32 hours.

### **Elite Hackathon Placements**
* 🏆 **4th / 150+ (HackUPC 2025):** Built **[OnlyFly](https://github.com/Blackhand01/HackUPC-Spring2025)**, a sustainable AI home-exchange MVP matching preferences and minimizing trip costs via Skyscanner/Revolut APIs. (36 hours).
* 🏆 **7th / 4,500 (GenAI.Works 2024):** Built **[PostGenius](https://github.com/Blackhand01/PostGenius)**, a robust RAG-based multimedia engine parsing news into hallucination-controlled, multi-platform posts via Vectara and Runway.

---

## ⚙️ Core Stack & Skills

* **Systems & Compute:** Python, C/C++, Rust, CUDA (Core concepts), PyTorch, JAX, Apple Silicon/MLX.
* **Edge & Robotics:** ROS2, MuJoCo, Gymnasium, PX4/MAVSDK, NVIDIA Jetson, Arduino Nicla Vision.
* **AI Architecture:** Inference Optimization, Model Quantization, Vision-Language-Action (VLA), LLM Agents, Diffusion Models, SSL.
* **Infra & MLOps:** Docker, Kubernetes, CI/CD, Triton Inference Server, FastApi, Telemetry & Tracing.

---

## 🎓 Education

* **M.Sc. in Artificial Intelligence and Data Analytics** | *Politecnico di Torino* (2024 – Expected 2026)
* **B.Sc. in Computer Engineering** | *Politecnico di Torino* (2020 – 2024)

---

## 📬 Contact

* 📧 **Email:** [bisiwork01@gmail.com](mailto:bisiwork01@gmail.com)
* 🔗 **LinkedIn:** [linkedin.com/in/stefanobisignano](https://www.linkedin.com/in/stefano-roy-bisignano-9100291b2/)
---