# Valora: Agentic AI Economic Simulation

Valora is a teaching/demo project that simulates a small economy with AI-driven actors. It blends machine learning “brains,” CrewAI orchestration, and a lightweight Flask API to explore how consumers, firms, banks, and regulators react to changing macro conditions and policy shocks.

## What this project does
- **Economic sandbox**: Steps through business cycles (expansion → peak → recession → trough) while tracking GDP, inflation, unemployment, and consumer confidence.
- **AI agents**: Each actor type uses a different brain (Q-learning, heuristics, RandomForest, PPO) to choose spending, pricing, lending, or taxation actions.
- **Policy experimentation**: Post shocks (e.g., financial crisis, pandemic) or submit policy proposals that get analyzed with macro context, debated, approved/rejected, and—if consensus is reached—applied for a configurable duration with traceable, revocable effects.
- **CrewAI pipeline**: Six agents run sequentially — the economic analyst reports conditions, the tax advisor suggests fiscal moves, the policy supporter argues pros, the policy opposer returns exactly four lines of quantified risks/mitigations, the policy critic independently issues a four-line verdict, and the blockchain expert audits pending transactions. Each agent response is capped at 300 words to respect model limits.
- **Blockchain-style logging**: Pending transactions are validated, hashed, and mined into an in-memory ledger with Merkle-style roots, audit history, and REST endpoints to inspect pending items, blocks, and chain integrity.
- **Fallback-friendly**: If you do not supply a Groq API key, the app continues with stub LLM responses so you can explore the simulation locally.

## Core components
- **`crew_orchestrator.py`**: Entrypoint that initializes the LLM (Groq or stub), sets up Flask routes, and starts background loops for economic updates and block mining.
- **Economic cycle manager**: Tracks the current cycle, rotates through phases, and applies immediate effects for shocks like `financial_crisis` or `pandemic`.
- **Agent brains**:
  - *Consumer*: Discrete Q-learning over inflation/savings buckets to pick spending fractions (0.3–0.9).
  - *Firm*: Heuristic controller for pricing/production with a placeholder for future DQN training.
  - *Bank*: RandomForest classifier trained on synthetic credit data to approve/deny loans.
  - *Regulator*: PPO agent (Stable Baselines3) that nudges tax rates toward inflation/unemployment targets.
- **Tax utility**: Simple order tax calculator with adjustable per-order-type modifiers.
- **Blockchain integration**: In-memory ledger plus pending transaction buffer; mining produces hashed blocks with Merkle-style roots, and APIs expose pending transactions, individual blocks, the full ledger, and verification/audit status.
- **Templates**: Basic HTML view rendered at `/` for quick inspection; extensible for richer dashboards later.

## How the pieces work together
1. The Flask app boots and tries to construct a Groq-backed `LLM` via CrewAI. Missing or failing credentials cause an automatic fallback to a stub LLM.
2. Background threads run every minute to (a) step the economic cycle and (b) mine any pending blockchain transactions into a new block.
3. REST endpoints let you:
   - Inspect cycle status and indicators.
   - Submit economic shocks with type, magnitude, and duration.
   - Post orders for tax calculation.
   - Send policy proposals (`POST /api/policies`) with optional intensity/policy-type overrides and list prior submissions plus active policy effects (`GET /api/policies`, `/api/policies/active`). Inspect or revoke a specific proposal via `GET /api/policies/<id>` and `POST /api/policies/<id>/revoke`. Proposals are analyzed with current macro context, debated, and—on consensus—applied as labeled, time-bounded policy shocks.
4. CrewAI agents operate in sequence: the economic analyst reviews indicators, the tax advisor proposes adjustments, policy supporter/opposer articulate pro and con cases, a policy critic independently issues a concise four-line conclusion, and the blockchain expert audits pending transactions. Recent runs are captured in memory and exposed via `/api/crew/runs` and on the dashboard.

## Running the project
Prerequisites: Python 3.10+, and (optionally) a Groq API key for richer LLM outputs.

```bash
# Install dependencies
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Run the interactive CLI (default)
python crew_orchestrator.py --mode cli

# Or start the Flask API + background loops
export GROQ_API_KEY=<your_key>   # optional; omit to use stub LLM
python crew_orchestrator.py --mode server
```

By default the server listens on port `5004` (override with `PORT`). Set `FLASK_SECRET_KEY` to customize the session secret, and `GROQ_MODEL` to choose a different Groq model. To further contain output size and token-per-minute usage, completions are capped with `MAX_AGENT_COMPLETION_TOKENS` (default 450 tokens) to keep every agent within the 300-word ceiling. The background threads that step the economic cycle and mine pending blockchain transactions start automatically when you launch the script.

## Extending and learning next
- **Improve learning**: Swap the firm heuristic for a DQN, add online updates for the bank model, or tune PPO reward shaping for the regulator.
- **Closer loop between AI outputs and state**: Map CrewAI recommendations into concrete parameter updates (e.g., dynamic tax multipliers or spending boosts).
- **Persistence and reproducibility**: Replace the in-memory ledger with a database and real hashing to preserve runs across restarts.
- **Frontend/UX**: Expand the `templates/` view or build a SPA that visualizes cycles, shocks, and agent actions in real time.
- **MLOps**: Add evaluation/monitoring to track agent rewards, policy outcomes, and shock recovery scenarios.

Valora is intentionally approachable: you can run it without external keys, inspect agent logic directly, and incrementally add sophistication as you explore agentic economics.
