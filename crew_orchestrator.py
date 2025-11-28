#!/usr/bin/env python3
"""
Crew Orchestrator for Valora — CrewAI + Groq LLM integrated example.

Notes:
- Set environment variable GROQ_API_KEY before running.
- If your crewai.LLM expects a different kwarg name for the provider key
  (e.g. groq_api_key), switch in the 'init_llm' function below.
- Disable Flask auto reloader (use_reloader=False) to avoid debugpy threading issues.
"""

import argparse
import hashlib
import importlib.util
import io
import json
import os
import threading
import time
import traceback
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

from flask import Flask, request, jsonify, render_template, session, redirect, url_for, send_file


def _coerce_json_safe(value):
    """Recursively convert datetime and other non-JSON-native types to serializable values."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, list):
        return [_coerce_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {k: _coerce_json_safe(v) for k, v in value.items()}
    return value


from crewai import Agent, Task, Crew, Process, LLM
from agents.bank_brain import BankRiskBrain
from agents.consumer_brain import ConsumerQBrain
from agents.firm_brain import FirmHeuristicBrain
from agents.regulator_brain import RegulatorPPOBrain

try:
    from economic_council import EconomicCouncil
except Exception:
    class EconomicCouncil:
        """Structured fallback if a richer council isn't available.

        Produces deterministic, explainable outputs so the policy experimentation flow
        stays functional without external dependencies.
        """

        def __init__(self, llm):
            self.llm = llm

        def analyze_policy(self, text, *, context=None):
            context = context or {}
            indicators = context.get("indicators", {})
            inflation = indicators.get("inflation", 2.0)
            unemployment = indicators.get("unemployment", 5.0)
            cycle = context.get("cycle", "expansion")

            lowered = text.lower()
            policy_type = "fiscal_policy"
            magnitude = 0.01
            rationale = ""
            if any(k in lowered for k in ["tax", "revenue", "levy"]):
                policy_type = "fiscal_policy"
                magnitude = -0.01 if "cut" in lowered or "reduce" in lowered else 0.015
                rationale = "Tax lever targeted."
            elif any(k in lowered for k in ["rate", "interest", "monetary", "tighten", "ease"]):
                policy_type = "monetary_policy"
                magnitude = -0.02 if "cut" in lowered or "ease" in lowered else 0.02
                rationale = "Interest rate calibration."
            elif any(k in lowered for k in ["stimulus", "spending", "aid", "relief", "grant"]):
                policy_type = "fiscal_policy"
                magnitude = 0.025
                rationale = "Spending-focused stimulus."
            else:
                rationale = "General structural reform."

            # Macro-aware tuning: lean against overheating or stagnation
            if inflation > 6.0 and policy_type == "fiscal_policy" and magnitude > 0:
                magnitude *= 0.6  # avoid overheating when inflation is high
                rationale += " Adjusted down due to elevated inflation."
            if unemployment > 8.0 and magnitude < 0:
                magnitude *= 0.5  # avoid contraction in weak labor markets
                rationale += " Softened to protect labor markets."
            if cycle == "recession" and magnitude < 0:
                magnitude *= 0.7
                rationale += " Avoiding austerity during recession."

            return {
                "policy_type": policy_type,
                "parameters": {"intensity": round(magnitude, 4)},
                "summary": f"Heuristic analysis inferred {policy_type} with intensity {magnitude:+.3f}. {rationale}",
                "expected_impacts": {
                    "gdp_growth": magnitude * 1.2,
                    "inflation": magnitude * 0.8,
                    "unemployment": -magnitude * 0.6,
                    "confidence": magnitude * 10.0,
                },
                "risk_score": min(0.9, max(0.1, abs(magnitude) * 4)),
            }

        def debate_policy(self, analysis, *, context=None):
            context = context or {}
            intensity = analysis.get("parameters", {}).get("intensity", 0.0)
            risks = analysis.get("risk_score", abs(intensity))
            inflation = context.get("indicators", {}).get("inflation", 2.0)
            unemployment = context.get("indicators", {}).get("unemployment", 5.0)
            views = []
            if risks < 0.25:
                views.append({"agent": "stability_advisor", "opinion": "approve", "reason": "Low systemic risk"})
            else:
                views.append({"agent": "stability_advisor", "opinion": "monitor", "reason": "Moderate risk"})
            if intensity > 0:
                views.append({"agent": "growth_hawk", "opinion": "approve", "reason": "Pro-growth"})
            else:
                views.append({"agent": "price_dove", "opinion": "approve", "reason": "Disinflationary"})
            if inflation > 7.0:
                views.append({"agent": "inflation_guard", "opinion": "monitor", "reason": "Inflation elevated"})
            if unemployment > 8.0:
                views.append({"agent": "labor_voice", "opinion": "approve", "reason": "Labor slack supports stimulus"})
            return views

        def reach_consensus(self, debate, *, context=None):
            approvals = [d for d in debate if d.get("opinion") == "approve"]
            return {
                "consensus_reached": len(approvals) >= 1,
                "notes": "Consensus reached via heuristic quorum" if approvals else "Insufficient support",
            }


MODEL_ID = os.getenv("GROQ_MODEL", "groq/llama-3.1-8b-instant")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")  # must be set in env
PORT = int(os.getenv("PORT", 5004))
# Cap completions so each agent respects the 300-word output ceiling while staying under TPM limits.
MAX_AGENT_COMPLETION_TOKENS = int(os.getenv("MAX_AGENT_COMPLETION_TOKENS", "300"))


app = Flask(__name__, template_folder="templates")
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-123")
os.makedirs("templates", exist_ok=True)


SIMULATION_PATH = Path(__file__).parent / "simulation engine" / "simulation.py"
_simulation_module = None


def load_simulation_module():
    global _simulation_module
    if _simulation_module:
        return _simulation_module
    if not SIMULATION_PATH.exists():
        raise FileNotFoundError(f"Simulation module not found at {SIMULATION_PATH}")
    spec = importlib.util.spec_from_file_location("policy_simulation", SIMULATION_PATH)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _simulation_module = module
        return module
    raise ImportError("Unable to load simulation module")


def init_llm():
    """
    Best-effort LLM initializer. Different versions of crewai/litellm expect different kwarg names:
      - api_key
      - groq_api_key
    We'll try a couple of common options and fall back to a minimal local stub LLM.
    """
    if not GROQ_API_KEY:
        print("[WARN] GROQ_API_KEY is not set. Agents will use fallback stub responses.")
        return None 
    try:
        llm = LLM(model=MODEL_ID, api_key=GROQ_API_KEY, max_tokens=MAX_AGENT_COMPLETION_TOKENS)
        print("[LLM] Initialized with LLM(model=..., api_key=...)")
        return llm
    except Exception as e1:
        print("[LLM] api_key arg failed:", e1)


        llm = LLM(model=MODEL_ID, groq_api_key=GROQ_API_KEY, max_tokens=MAX_AGENT_COMPLETION_TOKENS)
        print("[LLM] Initialized with LLM(model=..., groq_api_key=...)")
        return llm
    except Exception as e2:
        print("[LLM] groq_api_key arg failed:", e2)


    try:
        llm = LLM(model=MODEL_ID, max_tokens=MAX_AGENT_COMPLETION_TOKENS)
        print("[LLM] Initialized with LLM(model=...) (no key arg — relying on library env behavior)")
        return llm
    except Exception as e3:
        print("[LLM] model-only init failed:", e3)


    return None


class FallbackLLM:
    def __init__(self):
        pass

    def chat(self, prompt: str, **kwargs):
        # Very simple canned reply — allows app to keep working even without LLM access
        return {"content": "LLM unavailable — this is a fallback response for development."}


llm_instance = init_llm()
if llm_instance is None:
    fallback_llm = FallbackLLM()
    economic_council = EconomicCouncil(fallback_llm)
else:
    economic_council = EconomicCouncil(llm_instance)


simulation_state: Dict[str, dict] = {}
agent_reports: Dict[str, dict] = {}
crew_run_history: List[dict] = []


def _ensure_session_state() -> str:
    """Guarantee a session id and backing simulation_state entry."""
    sid = session.get("session_id")
    if not sid or sid not in simulation_state:
        sid = str(uuid.uuid4())
        session["session_id"] = sid
        simulation_state[sid] = {"policies": [], "debates": []}
    return sid


class EconomicCycleManager:
    def __init__(self):
        self.current_cycle = "expansion"
        self.cycle_start_date = datetime.now()
        self.cycle_duration = 365
        self.economic_indicators = {
            "gdp": 1000.0,
            "inflation": 2.0,
            "unemployment": 5.0,
            "consumer_confidence": 70.0,
        }
        # phase-specific annualized drifts and durations (days)
        self.phase_order = ["expansion", "peak", "recession", "trough"]
        self.phase_durations = {
            "expansion": 365,
            "peak": 120,
            "recession": 240,
            "trough": 180,
        }
        # Rates are annualized: inflation/unemployment/confidence expressed as percentage or index points per year
        self.phase_trends = {
            "expansion": {"gdp_growth": 0.03, "inflation_change": 1.2, "unemployment_change": -0.8, "confidence_change": 6.0},
            "peak": {"gdp_growth": 0.01, "inflation_change": 0.8, "unemployment_change": 0.1, "confidence_change": -2.0},
            "recession": {"gdp_growth": -0.025, "inflation_change": -0.6, "unemployment_change": 1.2, "confidence_change": -8.0},
            "trough": {"gdp_growth": 0.008, "inflation_change": 0.3, "unemployment_change": -0.6, "confidence_change": 4.0},
        }

        self.active_shocks = []
        self.last_update = datetime.now()

    def _advance_cycle_if_needed(self, now: datetime):
        days_in_cycle = (now - self.cycle_start_date).days
        duration = self.phase_durations.get(self.current_cycle, self.cycle_duration)
        if days_in_cycle >= duration:
            idx = self.phase_order.index(self.current_cycle)
            self.current_cycle = self.phase_order[(idx + 1) % len(self.phase_order)]
            self.cycle_start_date = now

    def _shock_profile(self, shock_type: str, magnitude: float):
        base_profiles = {
            "financial_crisis": {"gdp_growth": -0.06, "inflation_change": -1.0, "unemployment_change": 2.5, "confidence_change": -15.0},
            "pandemic": {"gdp_growth": -0.05, "inflation_change": -0.5, "unemployment_change": 2.0, "confidence_change": -18.0},
            "energy_price_spike": {"gdp_growth": -0.02, "inflation_change": 2.5, "unemployment_change": 0.4, "confidence_change": -6.0},
            "tech_boom": {"gdp_growth": 0.05, "inflation_change": 0.3, "unemployment_change": -1.5, "confidence_change": 12.0},
        }
        profile = base_profiles.get(
            shock_type,
            {"gdp_growth": -0.01, "inflation_change": 0.0, "unemployment_change": 0.0, "confidence_change": -3.0},
        )
        return {k: v * magnitude for k, v in profile.items()}

    def apply_economic_shock(self, shock_type: str, magnitude: float, duration_days: int):
        profile = self._shock_profile(shock_type, magnitude)
        shock = {
            "type": shock_type,
            "magnitude": magnitude,
            "profile": profile,
            "start_time": datetime.now(),
            "end_time": datetime.now() + timedelta(days=duration_days),
        }
        self.active_shocks.append(shock)
        # immediate shock to confidence to reflect sentiment changes
        self.economic_indicators["consumer_confidence"] = max(
            0.0,
            min(100.0, self.economic_indicators["consumer_confidence"] + profile.get("confidence_change", 0.0)),
        )

    def apply_policy_adjustment(
        self,
        policy_type: str,
        magnitude: float,
        duration_days: int = 180,
        *,
        label: str = None,
        policy_id: str | None = None,
    ):
        """Injects temporary policy adjustments as managed shocks (positive or negative).

        Returns the shock record so callers can track applied effects.
        """
        mapped_type = {
            "fiscal_policy": "stimulus" if magnitude >= 0 else "austerity",
            "monetary_policy": "rate_change",
        }.get(policy_type, policy_type)

        # Stimulus boosts growth/confidence; austerity does the opposite
        if mapped_type == "stimulus":
            profile = {
                "gdp_growth": 0.02 * magnitude,
                "inflation_change": 1.0 * magnitude,
                "unemployment_change": -1.0 * magnitude,
                "confidence_change": 8.0 * magnitude,
            }
        elif mapped_type == "austerity":
            profile = {
                "gdp_growth": -0.015 * abs(magnitude),
                "inflation_change": -0.8 * abs(magnitude),
                "unemployment_change": 0.8 * abs(magnitude),
                "confidence_change": -6.0 * abs(magnitude),
            }
        elif mapped_type == "rate_change":
            profile = {
                "gdp_growth": -0.01 * magnitude,
                "inflation_change": -1.5 * magnitude,
                "unemployment_change": 0.4 * magnitude,
                "confidence_change": -3.0 * magnitude,
            }
        else:
            profile = {
                "gdp_growth": 0.005 * magnitude,
                "inflation_change": 0.2 * magnitude,
                "unemployment_change": -0.2 * magnitude,
                "confidence_change": 2.0 * magnitude,
            }

        shock = {
            "id": str(uuid.uuid4()),
            "type": f"policy:{mapped_type}",
            "magnitude": magnitude,
            "profile": profile,
            "start_time": datetime.now(),
            "end_time": datetime.now() + timedelta(days=duration_days),
            "label": label or mapped_type,
            "policy_id": policy_id,
        }
        self.active_shocks.append(shock)
        return shock

    def _aggregate_shock_effects(self, now: datetime):
        effects = {"gdp_growth": 0.0, "inflation_change": 0.0, "unemployment_change": 0.0, "confidence_change": 0.0}
        still_active = []
        for shock in self.active_shocks:
            if shock["end_time"] >= now:
                for k in effects:
                    effects[k] += shock["profile"].get(k, 0.0)
                still_active.append(shock)
        self.active_shocks = still_active
        return effects

    def step(self):
        now = datetime.now()
        delta_days = max((now - self.last_update).total_seconds() / 86400.0, 0)
        delta_years = delta_days / 365.0
        self.last_update = now
        self._advance_cycle_if_needed(now)

        # Base trends for the current phase
        trends = self.phase_trends.get(self.current_cycle, self.phase_trends["expansion"])
        shock_effects = self._aggregate_shock_effects(now)

        gdp_growth = trends["gdp_growth"] + shock_effects["gdp_growth"]
        inflation_change = trends["inflation_change"] + shock_effects["inflation_change"]
        unemployment_change = trends["unemployment_change"] + shock_effects["unemployment_change"]
        confidence_change = trends["confidence_change"] + shock_effects["confidence_change"]

        # Apply changes scaled by elapsed time
        self.economic_indicators["gdp"] *= (1 + gdp_growth * delta_years)
        self.economic_indicators["inflation"] = max(
            0.0, self.economic_indicators["inflation"] + inflation_change * delta_years
        )
        self.economic_indicators["unemployment"] = max(
            0.0, min(35.0, self.economic_indicators["unemployment"] + unemployment_change * delta_years)
        )
        self.economic_indicators["consumer_confidence"] = max(
            0.0, min(100.0, self.economic_indicators["consumer_confidence"] + confidence_change * delta_years)
        )

    def get_status(self):
        self.step()
        return {
            "current_cycle": self.current_cycle,
            "days_in_cycle": (datetime.now() - self.cycle_start_date).days,
            "indicators": self.economic_indicators,
            "active_shocks": [
                {
                    "type": shock["type"],
                    "magnitude": shock["magnitude"],
                    "ends_in_days": max((shock["end_time"] - datetime.now()).days, 0),
                }
                for shock in self.active_shocks
            ],
        }

class OrderTaxAdjustment:
    def __init__(self):
        self.base_tax_rate = 0.10
        self.adjustments = {}

    def calculate_tax(self, order_value: float, order_type: str = None):
        rate = self.base_tax_rate
        if order_type and order_type in self.adjustments:
            rate += self.adjustments[order_type]
        return order_value * rate

class BlockchainIntegration:
    def __init__(self):
        self.ledger: List[dict] = []
        self.pending_transactions: List[dict] = []
        self.audit_log: List[dict] = []
        self.lock = threading.Lock()

    def _hash_json(self, data: dict | list) -> str:
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def _merkle_root(self, txs: List[dict]) -> str:
        if not txs:
            return self._hash_json([])
        layer = [self._hash_json(tx) for tx in txs]
        while len(layer) > 1:
            next_layer = []
            for i in range(0, len(layer), 2):
                left = layer[i]
                right = layer[i + 1] if i + 1 < len(layer) else layer[i]
                next_layer.append(self._hash_json({"left": left, "right": right}))
            layer = next_layer
        return layer[0]

    def _validate_transaction(self, tx: dict) -> dict:
        if not isinstance(tx, dict):
            raise ValueError("Transaction must be a JSON object")
        required_fields = ["from", "to", "amount"]
        missing = [f for f in required_fields if f not in tx]
        if missing:
            raise ValueError(f"Missing transaction fields: {', '.join(missing)}")
        amount = float(tx["amount"])
        if amount <= 0:
            raise ValueError("Transaction amount must be positive")
        sanitized = {
            "id": tx.get("id") or str(uuid.uuid4()),
            "from": str(tx["from"]),
            "to": str(tx["to"]),
            "amount": amount,
            "memo": tx.get("memo", ""),
        }
        return sanitized

    def add_transaction(self, tx: dict):
        sanitized = self._validate_transaction(tx)
        sanitized["timestamp"] = datetime.now().isoformat()
        sanitized["status"] = "pending"
        sanitized["hash"] = self._hash_json({k: sanitized[k] for k in sorted(sanitized) if k != "status"})
        with self.lock:
            self.pending_transactions.append(sanitized)
        return sanitized

    def _build_block(self, txs: List[dict]) -> dict:
        previous_hash = self.ledger[-1]["hash"] if self.ledger else "0"
        index = len(self.ledger) + 1
        timestamp = datetime.now().isoformat()
        merkle_root = self._merkle_root(txs)
        block_core = {
            "index": index,
            "timestamp": timestamp,
            "previous_hash": previous_hash,
            "merkle_root": merkle_root,
            "transactions": txs,
        }
        block_hash = self._hash_json(block_core)
        block_core["hash"] = block_hash
        return block_core

    def mine_block(self):
        with self.lock:
            if not self.pending_transactions:
                return None
            txs = []
            for tx in self.pending_transactions:
                tx_confirmed = dict(tx)
                tx_confirmed["status"] = "confirmed"
                txs.append(tx_confirmed)
            block = self._build_block(txs)
            self.ledger.append(block)
            self.pending_transactions = []
            self.audit_log.append(
                {
                    "timestamp": block["timestamp"],
                    "action": "mined",
                    "block_index": block["index"],
                    "hash": block["hash"],
                    "tx_count": len(txs),
                }
            )
            return block

    def verify_chain(self) -> Tuple[bool, List[str]]:
        errors = []
        with self.lock:
            for i, block in enumerate(self.ledger):
                expected_prev = "0" if i == 0 else self.ledger[i - 1]["hash"]
                if block.get("previous_hash") != expected_prev:
                    errors.append(f"Block {block.get('index')} previous_hash mismatch")
                recalculated = self._hash_json({
                    k: block[k]
                    for k in ["index", "timestamp", "previous_hash", "merkle_root", "transactions"]
                })
                if block.get("hash") != recalculated:
                    errors.append(f"Block {block.get('index')} hash invalid")
        return len(errors) == 0, errors

    def get_block(self, index: int):
        with self.lock:
            if 1 <= index <= len(self.ledger):
                return self.ledger[index - 1]
        return None

    def get_ledger(self, limit: int | None = None):
        with self.lock:
            data = list(self.ledger)
        if limit is not None and limit > 0:
            return data[-limit:]
        return data

    def get_pending(self):
        with self.lock:
            return list(self.pending_transactions)

    def get_audit_log(self, limit: int | None = None):
        with self.lock:
            data = list(self.audit_log)
        if limit is not None and limit > 0:
            return data[-limit:]
        return data

class PolicyExperimentationManager:
    """Coordinates policy proposals, analysis, consensus, and application."""

    def __init__(self, econ: EconomicCycleManager, council: EconomicCouncil):
        self.econ = econ
        self.council = council
        self.proposals = []

    def _current_context(self):
        return {
            "cycle": self.econ.current_cycle,
            "indicators": dict(self.econ.economic_indicators),
            "active_shocks": [s.copy() for s in self.econ.active_shocks],
        }

    def _call_council(self, method_name: str, *args, **kwargs):
        method = getattr(self.council, method_name)
        try:
            return method(*args, **kwargs)
        except TypeError:
            # fallback for councils that do not accept contextual kwargs
            return method(*args)

    def _normalize_adjustments(
        self,
        analysis: dict,
        default_duration: int,
        *,
        intensity_override: float | None = None,
        duration_override: int | None = None,
        policy_type_override: str | None = None,
    ):
        params = analysis.get("parameters", {}) if analysis else {}
        # prefer explicit intensity, fall back to spending_change or generic tweaks
        intensity = intensity_override
        if intensity is None:
            intensity = float(params.get("intensity", params.get("spending_change", 0.0)))
        policy_type = policy_type_override or (analysis.get("policy_type", "fiscal_policy") if analysis else "fiscal_policy")
        duration = duration_override if duration_override is not None else (
            int(params.get("duration_days", default_duration)) if params else default_duration
        )
        duration = max(7, min(365, duration))  # keep durations in a sensible window
        return [{"policy_type": policy_type, "magnitude": intensity, "duration_days": duration}]

    def _apply_adjustments(self, adjustments, *, label: str, policy_id: str):
        applied = []
        for adj in adjustments:
            shock = self.econ.apply_policy_adjustment(
                adj["policy_type"],
                adj.get("magnitude", 0.0),
                adj.get("duration_days", 180),
                label=label,
                policy_id=policy_id,
            )
            applied.append(shock)
        return applied

    def _validate_inputs(self, text: str, magnitude_override: float | None, duration_override: int | None):
        if not text or not text.strip():
            raise ValueError("Policy text is required")
        if magnitude_override is not None and abs(magnitude_override) > 5.0:
            raise ValueError("Magnitude override is unrealistically high; please stay within +/-5.0")
        if duration_override is not None and duration_override <= 0:
            raise ValueError("Duration override must be positive")

    def submit_policy(
        self,
        text: str,
        *,
        author: str = "unknown",
        source: str = "api",
        default_duration: int = 180,
        intensity_override: float | None = None,
        duration_override: int | None = None,
        policy_type_override: str | None = None,
    ):
        self._validate_inputs(text, intensity_override, duration_override)
        policy_id = str(uuid.uuid4())
        context = _coerce_json_safe(self._current_context())

        try:
            analysis = self._call_council("analyze_policy", text, context=context)
        except Exception as e:
            analysis = {"error": f"analysis failed: {e}"}

        debate = []
        consensus = {"consensus_reached": False, "notes": "skipped"}
        if "error" not in analysis:
            try:
                debate = self._call_council("debate_policy", analysis, context=context)
                consensus = self._call_council("reach_consensus", debate, context=context)
            except Exception as e:
                consensus = {"consensus_reached": False, "notes": f"debate failed: {e}"}

        adjustments = []
        applied_effects = []
        status = "rejected"
        if consensus.get("consensus_reached") and "error" not in analysis:
            adjustments = self._normalize_adjustments(
                analysis,
                default_duration,
                intensity_override=intensity_override,
                duration_override=duration_override,
                policy_type_override=policy_type_override,
            )
            applied_effects = self._apply_adjustments(adjustments, label=text[:80], policy_id=policy_id)
            status = "applied"

        record = {
            "id": policy_id,
            "author": author,
            "source": source,
            "text": text,
            "context": context,
            "analysis": analysis,
            "debate": debate,
            "consensus": consensus,
            "adjustments": adjustments,
            "applied_effects": applied_effects,
            "status": status,
            "created_at": datetime.now().isoformat(),
        }
        serializable_record = _coerce_json_safe(record)
        self.proposals.append(serializable_record)
        return serializable_record

    def list_policies(self):
        return self.proposals

    def get_policy(self, policy_id: str):
        for p in self.proposals:
            if p.get("id") == policy_id:
                return p
        return None

    def revoke_policy(self, policy_id: str):
        removed = []
        remaining_shocks = []
        for shock in self.econ.active_shocks:
            if shock.get("policy_id") == policy_id:
                removed.append(shock)
            else:
                remaining_shocks.append(shock)
        self.econ.active_shocks = remaining_shocks
        for record in self.proposals:
            if record.get("id") == policy_id:
                record["status"] = "revoked"
                record.setdefault("revoked_at", datetime.now().isoformat())
        return removed

    def active_policy_effects(self):
        now = datetime.now()
        return [
            {
                "id": shock.get("id"),
                "policy_id": shock.get("policy_id"),
                "label": shock.get("label"),
                "type": shock.get("type"),
                "magnitude": shock.get("magnitude"),
                "ends_in_days": max((shock.get("end_time") - now).days, 0)
                if shock.get("end_time")
                else None,
            }
            for shock in self.econ.active_shocks
            if shock.get("type", "").startswith("policy:")
        ]


economic_manager = EconomicCycleManager()
tax_manager = OrderTaxAdjustment()
blockchain = BlockchainIntegration()
policy_manager = PolicyExperimentationManager(economic_manager, economic_council)


class HeterogeneousAgentSimulation:
    """Connect consumer, firm, bank, and regulator brains to the economic sandbox."""

    def __init__(self, econ: EconomicCycleManager, taxes: OrderTaxAdjustment, seed: int = 2):
        self.econ = econ
        self.taxes = taxes

        self.consumer = ConsumerQBrain(seed=seed)
        self.firm = FirmHeuristicBrain()
        self.bank = BankRiskBrain(seed=seed)
        self.regulator = RegulatorPPOBrain()

        self.consumer_income = 120.0
        self.consumer_savings = 500.0
        self.baseline_spending = 140.0
        self.last_actions = {}

    def _apply_regulator_tax(self, delta_tax: float):
        base = self.taxes.base_tax_rate
        # keep tax rate in a realistic corridor
        self.taxes.base_tax_rate = max(0.01, min(0.35, base + delta_tax))
        self.last_actions["tax_rate"] = round(self.taxes.base_tax_rate, 4)

    def _update_economy_from_activity(self, spending: float, revenue: float, demand_signal: float):
        econ = self.econ.economic_indicators
        # modest boost from aggregate demand
        econ["gdp"] += (spending + revenue) / 500.0
        econ["consumer_confidence"] = max(0.0, min(100.0, econ["consumer_confidence"] + (demand_signal - 1.0) * 1.5))
        # unemployment drifts toward a target implied by demand
        target_unemp = max(2.5, 12.0 - demand_signal * 5.0)
        econ["unemployment"] += (target_unemp - econ["unemployment"]) * 0.05
        # price pressure from high demand nudges inflation
        econ["inflation"] = max(0.0, econ["inflation"] + (demand_signal - 1.0) * 0.05)

    def step(self):
        econ = self.econ.economic_indicators
        inflation_rate = econ["inflation"] / 100.0

        # Consumer chooses spending share based on inflation + savings
        spend_frac = float(self.consumer.act({"inflation": econ["inflation"], "savings": self.consumer_savings}))
        consumer_spending = spend_frac * self.consumer_income
        self.consumer_savings = max(0.0, self.consumer_savings + self.consumer_income - consumer_spending)

        demand_signal = max(0.4, consumer_spending / self.baseline_spending)
        firm_action = self.firm.act({"demand_signal": demand_signal})
        firm_price = firm_action["price"]
        firm_output = firm_action["production"]
        realized_sales = min(firm_output, demand_signal * self.firm.capacity)
        revenue = firm_price * realized_sales

        # Bank evaluates creditworthiness; approve loan if savings depleted
        bank_decision = self.bank.act({
            "savings": self.consumer_savings,
            "income": self.consumer_income,
            "inflation": inflation_rate,
        })
        loan_disbursed = 0.0
        if bank_decision == "approve" and self.consumer_savings < 200:
            loan_disbursed = 250.0
            self.consumer_savings += loan_disbursed

        regulator_delta = float(self.regulator.act({
            "gdp_norm": econ["gdp"] / 1000.0,
            "inflation": inflation_rate,
            "unemployment": econ["unemployment"] / 100.0,
        }))
        self._apply_regulator_tax(regulator_delta)

        # Learning feedback for consumer only (others are heuristic/static)
        reward = (consumer_spending / max(self.consumer_income, 1)) - inflation_rate - (econ["unemployment"] / 100)
        self.consumer.learn(reward, {"inflation": econ["inflation"], "savings": self.consumer_savings})

        self._update_economy_from_activity(consumer_spending + loan_disbursed, revenue, demand_signal)

        self.last_actions.update({
            "consumer_spending": round(consumer_spending, 2),
            "consumer_savings": round(self.consumer_savings, 2),
            "firm_price": round(firm_price, 2),
            "firm_output": firm_output,
            "bank_decision": bank_decision,
            "loan_disbursed": round(loan_disbursed, 2),
        })

    def status(self):
        return {
            "consumer": {
                "income": self.consumer_income,
                "savings": round(self.consumer_savings, 2),
            },
            "firm": {
                "last_price": self.last_actions.get("firm_price", self.firm.price),
                "capacity": self.firm.capacity,
            },
            "bank": {
                "last_decision": self.last_actions.get("bank_decision", "n/a"),
                "loan_disbursed": self.last_actions.get("loan_disbursed", 0.0),
            },
            "regulator": {
                "base_tax_rate": self.taxes.base_tax_rate,
            },
            "latest_actions": self.last_actions,
        }


agent_simulation = HeterogeneousAgentSimulation(economic_manager, tax_manager)


agent_llm = llm_instance if llm_instance is not None else FallbackLLM()

economic_analyst = Agent(
    role="Economic Analyst",
    goal="Analyze economic indicators and suggest policy actions",
    backstory="Economist with macro and policy expertise",
    llm=agent_llm,
    verbose=False
)

tax_advisor = Agent(
    role="Tax Advisor",
    goal="Recommend tax or fiscal adjustments given economic state",
    backstory="Tax expert focusing on efficient fiscal policy",
    llm=agent_llm,
    verbose=False
)

policy_supporter = Agent(
    role="Policy Supporter",
    goal="Champion proposed policies and forecast positive outcomes",
    backstory="Advocate who highlights benefits, confidence effects, and growth pathways",
    llm=agent_llm,
    verbose=False,
)

policy_opposer = Agent(
    role="Policy Opposer",
    goal="Stress-test policies by outlining risks and potential downsides with a clear oppose stance",
    backstory=(
        "Skeptical analyst focused on unintended consequences and fiscal discipline. "
        "Always take an oppositional view; never endorse the proposed policy."
    ),
    llm=agent_llm,
    verbose=False,
)

policy_critic = Agent(
    role="Policy Critic",
    goal="Synthesize pro and con perspectives into a balanced conclusion",
    backstory="Neutral reviewer who reconciles competing arguments into clear guidance",
    llm=agent_llm,
    verbose=False,
)

def blockchain_audit_tool():
    """Return a deterministic blockchain audit snapshot for the Crew."""
    validity, errors = blockchain.verify_chain()
    last_block = blockchain.ledger[-1] if blockchain.ledger else None
    return {
        "pending_count": len(blockchain.pending_transactions),
        "ledger_height": len(blockchain.ledger),
        "tip_hash": last_block.get("hash") if last_block else "genesis",
        "last_index": last_block.get("index") if last_block else 0,
        "valid": validity,
        "errors": errors,
        "recent_audit": blockchain.get_audit_log(10),
    }


blockchain_expert = Agent(
    role="Blockchain Specialist",
    goal="Validate and log transactions to ledger; audit blocks",
    backstory="Blockchain engineer and audit specialist",
    llm=agent_llm,
    verbose=False,
)

def build_crew(*, policy_text: str = "", include_blockchain: bool = True):
    indicators = economic_manager.economic_indicators
    cycle = economic_manager.current_cycle
    shocks = economic_manager.active_shocks
    policy_effects = policy_manager.active_policy_effects()
    policy_context = policy_text or "No policy provided; default to current stance."

    economic_task = Task(
        description=(
            "Analyze the current macro indicators and provide a concise report with 3 bullet insights and 1 recommended policy action. "
            f"GDP: {indicators['gdp']:.2f}, inflation: {indicators['inflation']:.2f}%, unemployment: {indicators['unemployment']:.2f}%, "
            f"consumer confidence: {indicators['consumer_confidence']:.2f}. Cycle phase: {cycle}. "
            f"Active shocks: {shocks}. Active policy effects: {policy_effects}. "
            f"Proposed policy under review: {policy_context}. "
            "Keep the entire response within 300 words."
        ),
        expected_output="short_report",
        agent=economic_analyst,
    )

    tax_task = Task(
        description=(
            "Given the latest macro indicators, recommend tax or fiscal adjustments with pros/cons. "
            f"Input indicators — GDP: {indicators['gdp']:.2f}, inflation: {indicators['inflation']:.2f}%, unemployment: {indicators['unemployment']:.2f}%. "
            f"Policy to consider: {policy_context}. "
            "Keep the entire response within 300 words."
        ),
        expected_output="tax_recommendations",
        agent=tax_advisor,
        dependencies=[economic_task],
    )

    supporter_task = Task(
        description=(
            "Review the economic analysis and tax recommendations, then argue in support of the policy direction. "
            f"Proposed policy text: {policy_context}. "
            "Explain why the approach should work, the channels of impact, and the expected positive outcomes. "
            "Keep the entire response within 300 words."
        ),
        expected_output="policy_support",
        agent=policy_supporter,
        dependencies=[tax_task],
    )

    opposer_task = Task(
        description=(
            "Critically oppose the current policy approach by identifying risks, trade-offs, and failure modes. "
            "Ground the objections in the provided economic analysis and tax advice. "
            "Return exactly four lines total: line 1 is a one-line verdict starting with 'Oppose'. Lines 2-4 each "
            "capture one quantified risk/failure mode with a mitigation, formatted as 'Risk: <metric impact>; Mitigation: "
            "<action>'. Keep the whole reply under 120 words; do not add headings, bullets, or praise/endorsement. "
            f"Policy text to challenge: {policy_context}. "
            "Keep the entire response within 300 words."
        ),
        expected_output="policy_opposition",
        agent=policy_opposer,
        dependencies=[tax_task],
    )

    critic_task = Task(
        description=(
            "Independently judge the policy direction without reading supporter/opposer outputs. "
            "Produce exactly four plain-text lines with no headers or bullets: line 1 is the verdict as Proceed/Modify/Pause;"
            " lines 2-4 are three brief reasons anchored in macro indicators and tax advice. Keep it under 80 words total. "
            f"Policy being assessed: {policy_context}. "
            "Keep the entire response within 300 words."
        ),
        expected_output="policy_conclusion",
        agent=policy_critic,
    )

    agents = [
        economic_analyst,
        tax_advisor,
        policy_supporter,
        policy_opposer,
        policy_critic,
    ]
    tasks = [
        economic_task,
        tax_task,
        supporter_task,
        opposer_task,
        critic_task,
    ]

    if include_blockchain:
        pending = len(blockchain.pending_transactions)
        ledger_height = len(blockchain.ledger)
        last_block = blockchain.ledger[-1] if blockchain.ledger else None
        last_hash = last_block.get("hash") if last_block else "genesis"
        audit_task = Task(
            description=(
                "Audit pending transactions and confirm integrity of ledger; list pending tx count and any hash inconsistencies. "
                f"Pending transactions: {pending}, ledger height: {ledger_height}, tip hash: {last_hash}. "
                "Keep the entire response within 300 words."
            ),
            expected_output="audit_report",
            agent=blockchain_expert,
            dependencies=[critic_task],
        )
        agents.append(blockchain_expert)
        tasks.append(audit_task)

    return Crew(
        agents=agents,
        tasks=tasks,
        verbose=True,
        process=Process.sequential,
        manager_llm=agent_llm,
    )


@app.route("/api/economic/status", methods=["GET"])
def api_economic_status():
    return jsonify(economic_manager.get_status())


@app.route("/api/agents/status", methods=["GET"])
def api_agents_status():
    return jsonify(agent_simulation.status())


def _stringify_task_output(output: Any) -> str:
    if output is None:
        return "No output recorded."
    if isinstance(output, str):
        return output
    try:
        return json.dumps(output, indent=2, default=str)
    except Exception:
        return str(output)


@app.route("/api/agents/run", methods=["POST"])
def api_agents_run():
    sid = _ensure_session_state()
    data = request.get_json(force=True)
    policy_text = (data.get("text") or data.get("policy_text") or "").strip()
    if not policy_text:
        return jsonify({"error": "policy text is required"}), 400

    crew = build_crew(policy_text=policy_text, include_blockchain=False)
    try:
        final_output = crew.kickoff()
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Crew run failed: {exc}"}), 500

    agent_outputs = []
    for task in crew.tasks:
        agent_outputs.append({
            "agent": getattr(task.agent, "role", "agent"),
            "task": task.description,
            "output": _stringify_task_output(getattr(task, "output", None)),
        })

    agent_reports[sid] = {
        "policy_text": policy_text,
        "agents": agent_outputs,
        "final_output": _stringify_task_output(final_output),
    }

    return jsonify({
        "policy": policy_text,
        "final_output": _stringify_task_output(final_output),
        "agents": agent_outputs,
    })


@app.route("/api/agents/report", methods=["GET"])
def api_agents_report():
    sid = session.get("session_id")
    run = agent_reports.get(sid)
    if not run:
        return jsonify({"error": "no agent run found for this session"}), 404

    buffer = io.StringIO()
    buffer.write(f"Policy: {run['policy_text']}\n\n")
    for idx, entry in enumerate(run.get("agents", []), start=1):
        buffer.write(f"{idx}. {entry.get('agent', 'Agent')}\n")
        buffer.write(f"{entry.get('output', '').strip()}\n\n")
    buffer.write("Final Output:\n")
    buffer.write(run.get("final_output", "") + "\n")
    payload = io.BytesIO(buffer.getvalue().encode("utf-8"))
    payload.seek(0)
    return send_file(payload, as_attachment=True, download_name="policy_report.txt", mimetype="text/plain")


@app.route("/api/policies", methods=["GET"])
def api_policies_list():
    return jsonify({"policies": policy_manager.list_policies(), "active_effects": policy_manager.active_policy_effects()})


@app.route("/api/policies", methods=["POST"])
def api_submit_policy():
    data = request.get_json(force=True)
    text = data.get("text") or data.get("policy_text") or ""
    author = data.get("author", "api")
    duration = int(data.get("duration_days", 180))
    intensity = data.get("intensity")
    policy_type_override = data.get("policy_type")
    intensity_override = float(intensity) if intensity is not None else None
    try:
        record = policy_manager.submit_policy(
            text,
            author=author,
            source="api",
            default_duration=duration,
            intensity_override=intensity_override,
            policy_type_override=policy_type_override,
        )
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    return jsonify({
        "status": record["status"],
        "policy": record,
        "active_effects": policy_manager.active_policy_effects(),
        "indicators": economic_manager.economic_indicators,
    })


@app.route("/api/policies/<policy_id>", methods=["GET"])
def api_policy_detail(policy_id):
    policy = policy_manager.get_policy(policy_id)
    if not policy:
        return jsonify({"error": "policy not found"}), 404
    return jsonify(policy)


@app.route("/api/policies/<policy_id>/revoke", methods=["POST"])
def api_policy_revoke(policy_id):
    removed = policy_manager.revoke_policy(policy_id)
    if not removed:
        return jsonify({"status": "no_active_effects", "policy_id": policy_id})
    return jsonify({"status": "revoked", "policy_id": policy_id, "removed_effects": removed})


@app.route("/api/policies/active", methods=["GET"])
def api_policies_active():
    return jsonify({"active_effects": policy_manager.active_policy_effects()})

@app.route("/api/economic/shock", methods=["POST"])
def api_economic_shock():
    data = request.get_json(force=True)
    economic_manager.apply_economic_shock(
        data.get("type", "pandemic"),
        float(data.get("magnitude", 1.0)),
        int(data.get("duration_days", 90))
    )
    return jsonify({"status": "ok", "active_shocks": economic_manager.get_status().get("active_shocks", [])})

@app.route("/api/tax/calc", methods=["POST"])
def api_tax_calc():
    data = request.get_json(force=True)
    order_value = float(data.get("order_value", 0.0))
    order_type = data.get("order_type")
    tax = tax_manager.calculate_tax(order_value, order_type)
    return jsonify({"order_value": order_value, "tax": tax})

@app.route("/api/blockchain/tx", methods=["POST"])
def api_add_tx():
    data = request.get_json(force=True)
    try:
        tx = blockchain.add_transaction(data)
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    return jsonify({"status": "tx_added", "tx": tx, "pending": len(blockchain.pending_transactions)})

@app.route("/api/blockchain/mine", methods=["POST"])
def api_mine():
    block = blockchain.mine_block()
    if block:
        return jsonify({"status": "mined", "block": block})
    return jsonify({"status": "no_pending"})


@app.route("/api/blockchain/ledger", methods=["GET"])
def api_ledger():
    limit = request.args.get("limit")
    limit_int = int(limit) if limit else None
    return jsonify({"ledger": blockchain.get_ledger(limit_int)})


@app.route("/api/blockchain/pending", methods=["GET"])
def api_pending():
    return jsonify({"pending": blockchain.get_pending()})


@app.route("/api/blockchain/block/<int:index>", methods=["GET"])
def api_block_detail(index: int):
    block = blockchain.get_block(index)
    if not block:
        return jsonify({"error": "block_not_found"}), 404
    return jsonify(block)


@app.route("/api/blockchain/verify", methods=["GET"])
def api_verify_chain():
    ok, errors = blockchain.verify_chain()
    return jsonify({"valid": ok, "errors": errors, "audit": blockchain.get_audit_log(25)})


@app.route("/api/blockchain/audit", methods=["GET"])
def api_audit_log():
    limit = request.args.get("limit")
    limit_int = int(limit) if limit else None
    return jsonify({"audit": blockchain.get_audit_log(limit_int)})


@app.route("/api/crew/runs", methods=["GET"])
def api_crew_runs():
    limit = request.args.get("limit")
    limit_int = int(limit) if limit else 20
    return jsonify({"runs": crew_run_history[-limit_int:]})


@app.route("/api/simulation/run", methods=["POST"])
def run_policy_simulation():
    data = request.get_json(silent=True) or {}
    policy_text = (data.get("policy") or data.get("text") or "").strip()
    time_steps = int(data.get("time_steps") or 8)

    if not policy_text:
        return jsonify({"error": "Policy text is required."}), 400

    simulation_module = load_simulation_module()
    result = simulation_module.simulate_economy(policy_text, time_steps=time_steps)
    return jsonify(_coerce_json_safe(result))


@app.route("/simulation")
def simulation_page():
    return render_template("simulation.html")

@app.route("/")
def index():
    sid = _ensure_session_state()
    return render_template(
        "index.html",
        indicators=economic_manager.economic_indicators,
        policies=simulation_state[sid]["policies"],
        debates=simulation_state[sid]["debates"],
        crew_runs=crew_run_history[-5:],
        max_tokens=MAX_AGENT_COMPLETION_TOKENS,
    )

@app.route("/submit_policy", methods=["POST"])
def submit_policy():
    sid = _ensure_session_state()
    policy_text = request.form.get("policy_text", "")
    if not policy_text:
        return jsonify({"error": "no policy text"}), 400

    try:
        record = policy_manager.submit_policy(policy_text, author=sid, source="web")
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400

    simulation_state[sid]["policies"].append(record)
    simulation_state[sid]["debates"].append(record)
    return jsonify({
        "status": record.get("status"),
        "policy": record,
        "indicators": economic_manager.economic_indicators,
        "active_effects": policy_manager.active_policy_effects(),
    })

@app.route("/reset_session", methods=["POST"])
def reset_session():
    if "session_id" in session:
        sid = session.pop("session_id")
        simulation_state.pop(sid, None)
        agent_reports.pop(sid, None)
    return redirect(url_for("index"))


def econ_cycle_loop():
    while True:
        try:
            agent_simulation.step()
            economic_manager.step()
            # mine a block every minute (simple)
            blockchain.mine_block()
            time.sleep(60)
        except Exception:
            print("[ERROR] econ_cycle_loop", traceback.format_exc())
            time.sleep(5)

def crew_loop():
    while True:
        try:
            print(f"[{datetime.now().isoformat()}] Running Crew kickoff...")
            # Build a fresh crew each run to inject up-to-date context
            latest_policy_text = policy_manager.proposals[-1]["text"] if policy_manager.proposals else ""
            dynamic_crew = build_crew(policy_text=latest_policy_text)
            state_snapshot = {
                "timestamp": datetime.now().isoformat(),
                "indicators": economic_manager.economic_indicators.copy(),
                "pending_tx": len(blockchain.pending_transactions),
                "ledger_height": len(blockchain.ledger),
                "cycle": economic_manager.current_cycle,
            }
            try:
                res = dynamic_crew.kickoff()
                crew_run_history.append({"state": state_snapshot, "result": str(res)})
                # keep the history bounded
                if len(crew_run_history) > 25:
                    del crew_run_history[0]
                print("[Crew] kickoff completed.")
            except Exception as e:
                print("[Crew] kickoff failed:", repr(e))
                crew_run_history.append({"state": state_snapshot, "error": str(e)})
                # print minimal trace
                traceback.print_exc()
        except Exception as outer:
            print("[ERROR] crew_loop outer", traceback.format_exc())
        time.sleep(3600)  # run hourly

def run_cli_interface():
    print(
        "Valora CLI — submit policy proposals directly from the terminal. "
        "Type 'quit' to exit."
    )
    while True:
        try:
            policy_text = input("Enter a policy proposal: ").strip()
        except EOFError:
            print("\n[CLI] Input stream closed. Exiting.")
            break

        if not policy_text:
            print("[CLI] Please provide non-empty policy text or type 'quit' to exit.")
            continue
        if policy_text.lower() in {"quit", "exit"}:
            print("[CLI] Goodbye!")
            break

        try:
            record = policy_manager.submit_policy(policy_text, author="cli", source="cli")
        except ValueError as ve:
            print(f"[CLI] Policy submission error: {ve}")
            continue

        print("\n=== Policy Submission Recorded ===")
        print(json.dumps(record, indent=2))

        try:
            crew = build_crew(policy_text=policy_text)
            result = crew.kickoff()
            print("\n=== Crew Output ===")
            print(result)
        except Exception as e:
            print(f"[CLI] Crew run failed: {e}")

        econ_state = economic_manager.economic_indicators
        print("\n=== Updated Economic Indicators ===")
        print(json.dumps(econ_state, indent=2))


def run_server():
    # Background threads
    threading.Thread(target=econ_cycle_loop, daemon=True).start()
    threading.Thread(target=crew_loop, daemon=True).start()

    # Run Flask without the reloader to avoid debugpy/threading issues
    app.run(host="0.0.0.0", port=PORT, debug=True, use_reloader=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Valora orchestration entrypoint")
    parser.add_argument(
        "--mode",
        choices=["cli", "server"],
        default="server",
        help="Start the Flask server (default) or run the interactive CLI",
    )
    args = parser.parse_args()

    if args.mode == "server":
        run_server()
    else:
        run_cli_interface()


