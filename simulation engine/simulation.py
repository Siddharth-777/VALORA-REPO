from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


PolicySignals = Dict[str, float]
StateVector = Dict[str, float]


@dataclass
class ActorBlueprint:
    """Static description of a macro actor in the economy."""

    name: str
    sector: str
    keywords: List[str]
    base_state: StateVector
    behavioural_rules: Dict[str, float]
    reward_penalty: str


@dataclass
class SimulationAgent:
    """Runtime representation of an actor tailored to a specific policy."""

    name: str
    sector: str
    state: StateVector
    behavioural_rules: Dict[str, float]
    reward_penalty: str

    def step(self, policy_signals: PolicySignals, time_index: int) -> StateVector:
        """Advance the agent state by one time step based on policy signals."""

        fiscal_effect = policy_signals.get("fiscal", 0.0)
        monetary_effect = policy_signals.get("monetary", 0.0)
        regulatory_effect = policy_signals.get("regulation", 0.0)

        net_pressure = (
            fiscal_effect * self.behavioural_rules.get("fiscal_sensitivity", 0.0)
            + monetary_effect * self.behavioural_rules.get("monetary_sensitivity", 0.0)
            + regulatory_effect * self.behavioural_rules.get("regulatory_sensitivity", 0.0)
        )

        decay = max(0.3, 1.0 - 0.07 * time_index)
        directional_weight = self.behavioural_rules.get("policy_alignment", 1.0)
        adjustment_rate = self.behavioural_rules.get("adjustment_rate", 0.25)
        delta = net_pressure * adjustment_rate * decay * directional_weight

        for param, value in list(self.state.items()):
            sensitivity = self.behavioural_rules.get(f"{param}_sensitivity", 1.0)
            param_bias = self.behavioural_rules.get(f"{param}_bias", 0.0)
            updated = value + delta * sensitivity + 0.01 * param_bias
            self.state[param] = max(-5.0, min(5.0, round(updated, 3)))

        return dict(self.state)


def _actor_blueprints() -> List[ActorBlueprint]:
    return [
        ActorBlueprint(
            name="Households (General)",
            sector="households_general",
            keywords=["household", "consumer", "family", "income", "spending"],
            base_state={
                "income_distribution": 0.0,
                "avg_household_size": 3.2,
                "consumption_elasticity": 0.5,
                "unemployment_susceptibility": 0.4,
                "liquidity_buffer_pct": 0.2,
                "essential_spending_share": 0.65,
            },
            behavioural_rules={
                "fiscal_sensitivity": 0.8,
                "monetary_sensitivity": 0.25,
                "regulatory_sensitivity": 0.2,
                "policy_alignment": 1.0,
                "adjustment_rate": 0.35,
                "income_distribution_sensitivity": 1.2,
                "unemployment_susceptibility_sensitivity": 1.1,
            },
            reward_penalty="Rewarded when disposable income and job stability rise; penalized by cost shocks.",
        ),
        ActorBlueprint(
            name="Low-Income Households",
            sector="households_low_income",
            keywords=["low income", "poverty", "welfare", "benefit", "subsidy"],
            base_state={
                "poverty_risk": 0.6,
                "benefit_dependency_rate": 0.45,
                "price_sensitivity": 0.8,
                "informal_sector_participation": 0.3,
                "default_probability": 0.25,
            },
            behavioural_rules={
                "fiscal_sensitivity": 1.0,
                "monetary_sensitivity": 0.2,
                "regulatory_sensitivity": 0.15,
                "policy_alignment": 1.1,
                "adjustment_rate": 0.45,
                "poverty_risk_sensitivity": 1.3,
                "price_sensitivity_sensitivity": 1.1,
            },
            reward_penalty="Rewarded by transfers and subsidies; penalized by inflation or benefit cuts.",
        ),
        ActorBlueprint(
            name="High-Income Households",
            sector="households_high_income",
            keywords=["wealth", "luxury", "investment", "asset", "offshore"],
            base_state={
                "asset_ownership_ratio": 0.85,
                "luxury_spending_share": 0.25,
                "investment_propensity": 0.7,
                "offshore_saving_tendency": 0.35,
                "wealth_tax_exposure": 0.2,
            },
            behavioural_rules={
                "fiscal_sensitivity": 0.6,
                "monetary_sensitivity": 0.4,
                "regulatory_sensitivity": 0.3,
                "policy_alignment": 0.9,
                "adjustment_rate": 0.3,
                "investment_propensity_sensitivity": 1.2,
                "wealth_tax_exposure_sensitivity": 1.1,
            },
            reward_penalty="Rewarded by investment incentives; penalized by wealth and capital taxes.",
        ),
        ActorBlueprint(
            name="Small Firms",
            sector="firms_small",
            keywords=["small business", "sme", "credit", "startup", "compliance"],
            base_state={
                "unit_cost": 1.0,
                "fixed_cost": 0.4,
                "variable_cost": 0.6,
                "production_capacity": 0.5,
                "market_power": 0.2,
                "loan_dependency": 0.55,
                "time_to_scale_production": 0.4,
            },
            behavioural_rules={
                "fiscal_sensitivity": 0.7,
                "monetary_sensitivity": 0.5,
                "regulatory_sensitivity": 0.6,
                "policy_alignment": 1.0,
                "adjustment_rate": 0.4,
                "production_capacity_sensitivity": 1.2,
                "loan_dependency_sensitivity": 1.1,
            },
            reward_penalty="Rewarded by tax breaks and credit access; penalized by compliance costs.",
        ),
        ActorBlueprint(
            name="Medium Firms",
            sector="firms_medium",
            keywords=["mid-size", "factory", "labor", "inventory", "supply"],
            base_state={
                "capacity_utilization_rate": 0.6,
                "labor_intensity": 0.55,
                "cost_pass_through_rate": 0.4,
                "credit_demand": 0.5,
                "inventory_buffer": 0.45,
                "supply_chain_risk": 0.35,
            },
            behavioural_rules={
                "fiscal_sensitivity": 0.65,
                "monetary_sensitivity": 0.35,
                "regulatory_sensitivity": 0.4,
                "policy_alignment": 0.95,
                "adjustment_rate": 0.32,
                "capacity_utilization_rate_sensitivity": 1.1,
                "supply_chain_risk_sensitivity": 1.05,
            },
            reward_penalty="Rewarded by demand growth and stable credit; penalized by supply disruptions.",
        ),
        ActorBlueprint(
            name="Large Corporations",
            sector="corporate_large",
            keywords=["corporate", "multinational", "shareholder", "debt", "investment"],
            base_state={
                "capital_intensity": 0.75,
                "global_exposure": 0.55,
                "supply_chain_depth": 0.5,
                "pricing_strategy": 0.45,
                "shareholder_pressure": 0.6,
                "debt_maturity_profile": 0.5,
                "investment_rate": 0.55,
            },
            behavioural_rules={
                "fiscal_sensitivity": 0.55,
                "monetary_sensitivity": 0.45,
                "regulatory_sensitivity": 0.45,
                "policy_alignment": 0.9,
                "adjustment_rate": 0.28,
                "investment_rate_sensitivity": 1.15,
                "global_exposure_sensitivity": 1.05,
            },
            reward_penalty="Rewarded by predictable policy and incentives; penalized by regulatory drag.",
        ),
        ActorBlueprint(
            name="Banks (Commercial)",
            sector="banks_commercial",
            keywords=["bank", "loan", "credit", "capital", "npl"],
            base_state={
                "capital_adequacy_ratio": 0.12,
                "loan_portfolio_risk": 0.35,
                "liquidity_coverage_ratio": 1.05,
                "interest_margin": 0.03,
                "nonperforming_loans_ratio": 0.04,
                "lending_policy": 0.5,
            },
            behavioural_rules={
                "fiscal_sensitivity": 0.25,
                "monetary_sensitivity": 0.85,
                "regulatory_sensitivity": 0.5,
                "policy_alignment": 1.0,
                "adjustment_rate": 0.3,
                "loan_portfolio_risk_sensitivity": 1.2,
                "lending_policy_sensitivity": 1.1,
            },
            reward_penalty="Rewarded by lower funding costs and healthy credit demand; penalized by rising defaults.",
        ),
        ActorBlueprint(
            name="Microfinance Institutions",
            sector="microfinance",
            keywords=["microfinance", "rural", "women", "small loan", "cap"],
            base_state={
                "avg_loan_size": 0.002,
                "rural_penetration": 0.6,
                "default_rate_mfi": 0.08,
                "women_borrower_share": 0.7,
                "interest_rate_cap_policy_exposure": 0.5,
            },
            behavioural_rules={
                "fiscal_sensitivity": 0.45,
                "monetary_sensitivity": 0.55,
                "regulatory_sensitivity": 0.5,
                "policy_alignment": 1.05,
                "adjustment_rate": 0.33,
                "default_rate_mfi_sensitivity": 1.2,
                "rural_penetration_sensitivity": 1.1,
            },
            reward_penalty="Rewarded by concessional funding and targeted subsidies; penalized by tighter caps or rising defaults.",
        ),
        ActorBlueprint(
            name="Central Bank",
            sector="central_bank",
            keywords=["central bank", "policy rate", "reserve", "inflation", "liquidity"],
            base_state={
                "inflation_target": 0.05,
                "policy_rate": 0.04,
                "reserve_requirement": 0.09,
                "open_market_operations_capacity": 0.7,
                "communication_bias": 0.2,
                "currency_stability_priority": 0.6,
            },
            behavioural_rules={
                "fiscal_sensitivity": 0.2,
                "monetary_sensitivity": 1.0,
                "regulatory_sensitivity": 0.25,
                "policy_alignment": 1.0,
                "adjustment_rate": 0.27,
                "policy_rate_sensitivity": 1.25,
                "currency_stability_priority_sensitivity": 1.1,
            },
            reward_penalty="Rewarded for meeting inflation and stability goals; penalized by volatility and credibility loss.",
        ),
        ActorBlueprint(
            name="Government (Fiscal Authority)",
            sector="fiscal_authority",
            keywords=["budget", "spending", "tax", "fiscal", "transfer"],
            base_state={
                "fiscal_space": 0.35,
                "spending_priorities": 0.5,
                "tax_bases": 0.55,
                "budget_rigidity": 0.45,
                "automatic_stabilizers": 0.6,
                "transfer_payments_ratio": 0.4,
            },
            behavioural_rules={
                "fiscal_sensitivity": 1.1,
                "monetary_sensitivity": 0.2,
                "regulatory_sensitivity": 0.25,
                "policy_alignment": 1.0,
                "adjustment_rate": 0.3,
                "spending_priorities_sensitivity": 1.1,
                "transfer_payments_ratio_sensitivity": 1.15,
            },
            reward_penalty="Rewarded by revenue buoyancy and growth; penalized by fiscal stress and inefficiency.",
        ),
        ActorBlueprint(
            name="Tax Department / Revenue Authority",
            sector="tax_department",
            keywords=["tax authority", "compliance", "gst", "enforcement", "revenue"],
            base_state={
                "income_tax_enforcement_strength": 0.55,
                "corporate_tax_enforcement_strength": 0.5,
                "gst_compliance_rate": 0.65,
                "evasion_detection_prob": 0.4,
                "admin_efficiency": 0.5,
            },
            behavioural_rules={
                "fiscal_sensitivity": 0.95,
                "monetary_sensitivity": 0.1,
                "regulatory_sensitivity": 0.35,
                "policy_alignment": 1.0,
                "adjustment_rate": 0.34,
                "gst_compliance_rate_sensitivity": 1.1,
                "admin_efficiency_sensitivity": 1.05,
            },
            reward_penalty="Rewarded by higher compliance and efficiency; penalized by evasion and enforcement gaps.",
        ),
        ActorBlueprint(
            name="Regulatory Authority (Non-Monetary)",
            sector="regulator_non_monetary",
            keywords=["regulator", "compliance", "inspection", "penalty", "standard"],
            base_state={
                "regulatory_power_score": 0.5,
                "compliance_cost_factor": 0.45,
                "inspection_frequency": 0.5,
                "penalty_rate": 0.35,
                "sector_focus": 0.4,
            },
            behavioural_rules={
                "fiscal_sensitivity": 0.2,
                "monetary_sensitivity": 0.15,
                "regulatory_sensitivity": 1.0,
                "policy_alignment": 1.05,
                "adjustment_rate": 0.31,
                "inspection_frequency_sensitivity": 1.15,
                "compliance_cost_factor_sensitivity": 1.2,
            },
            reward_penalty="Rewarded by effective oversight; penalized when burdens hamper growth.",
        ),
        ActorBlueprint(
            name="Investors / Financial Markets",
            sector="financial_markets",
            keywords=["market", "investor", "portfolio", "liquidity", "risk"],
            base_state={
                "portfolio_allocation": 0.5,
                "risk_tolerance": 0.55,
                "reaction_speed": 0.6,
                "market_liquidity": 0.5,
                "foreign_flow_sensitivity": 0.45,
            },
            behavioural_rules={
                "fiscal_sensitivity": 0.45,
                "monetary_sensitivity": 0.55,
                "regulatory_sensitivity": 0.35,
                "policy_alignment": 1.05,
                "adjustment_rate": 0.36,
                "reaction_speed_sensitivity": 1.2,
                "market_liquidity_sensitivity": 1.1,
            },
            reward_penalty="Rewarded by clarity and liquidity; penalized by uncertainty and capital controls.",
        ),
        ActorBlueprint(
            name="Exporters / Trade Sector",
            sector="exporters",
            keywords=["export", "trade", "currency", "tariff", "shipping"],
            base_state={
                "export_volume": 0.5,
                "import_dependency": 0.45,
                "exchange_rate_exposure": 0.55,
                "global_demand_sensitivity": 0.6,
                "shipping_cost_index": 0.4,
            },
            behavioural_rules={
                "fiscal_sensitivity": 0.55,
                "monetary_sensitivity": 0.65,
                "regulatory_sensitivity": 0.35,
                "policy_alignment": 1.1,
                "adjustment_rate": 0.38,
                "exchange_rate_exposure_sensitivity": 1.2,
                "shipping_cost_index_sensitivity": 1.05,
            },
            reward_penalty="Rewarded by competitive exchange rates and logistics support; penalized by tariffs and high shipping costs.",
        ),
        ActorBlueprint(
            name="Energy Producers / Utility Sector",
            sector="energy_utilities",
            keywords=["energy", "utility", "fuel", "grid", "carbon", "kwh"],
            base_state={
                "generation_capacity": 0.55,
                "fuel_dependency_mix": 0.5,
                "cost_per_kwh": 0.12,
                "carbon_intensity": 0.4,
                "grid_stability_factor": 0.6,
                "energy_price_pass_through_rate": 0.5,
            },
            behavioural_rules={
                "fiscal_sensitivity": 0.4,
                "monetary_sensitivity": 0.25,
                "regulatory_sensitivity": 0.85,
                "policy_alignment": 1.05,
                "adjustment_rate": 0.37,
                "grid_stability_factor_sensitivity": 1.15,
                "cost_per_kwh_sensitivity": 1.2,
            },
            reward_penalty="Rewarded by investment support and stable regulation; penalized by emissions costs and input volatility.",
        ),
    ]


def extract_policy_signals(policy_text: str) -> PolicySignals:
    """Lightweight heuristic extraction of the policy's directional signals."""

    lowered = policy_text.lower()

    def _score(words: List[str]) -> int:
        return sum(lowered.count(w) for w in words)

    fiscal_words = ["tax", "spend", "stimulus", "subsidy", "grant", "budget", "rebate", "transfer"]
    monetary_words = ["rate", "interest", "liquidity", "reserve", "monetary", "credit", "loan"]
    regulatory_words = ["regulation", "standard", "compliance", "emission", "safety", "rule", "tariff", "inspection"]

    fiscal_score = _score(fiscal_words)
    monetary_score = _score(monetary_words)
    regulatory_score = _score(regulatory_words)

    direction = -1.0 if any(k in lowered for k in ["cut", "reduce", "ease"]) else 1.0

    fiscal_intensity = direction * min(1.0, 0.1 + 0.05 * fiscal_score)
    monetary_intensity = direction * min(1.0, 0.1 + 0.05 * monetary_score)
    regulatory_intensity = min(1.0, 0.1 + 0.05 * regulatory_score)

    return {
        "fiscal": round(fiscal_intensity, 3),
        "monetary": round(monetary_intensity, 3),
        "regulation": round(regulatory_intensity, 3),
    }


def select_affected_actors(policy_text: str, blueprints: List[ActorBlueprint], top_n: int = 5) -> List[ActorBlueprint]:
    lowered = policy_text.lower()

    def _score(bp: ActorBlueprint) -> Tuple[int, float]:
        keyword_score = sum(lowered.count(k) for k in bp.keywords)
        base_priority = bp.behavioural_rules.get("fiscal_sensitivity", 0) + bp.behavioural_rules.get(
            "monetary_sensitivity", 0
        ) + bp.behavioural_rules.get("regulatory_sensitivity", 0)
        return keyword_score, base_priority

    sorted_bps = sorted(blueprints, key=lambda bp: _score(bp), reverse=True)
    return sorted_bps[:top_n]


def build_agents_for_policy(policy_text: str) -> Tuple[List[SimulationAgent], PolicySignals]:
    blueprints = _actor_blueprints()
    policy_signals = extract_policy_signals(policy_text)
    selected_bps = select_affected_actors(policy_text, blueprints, top_n=5)

    agents = [
        SimulationAgent(
            name=bp.name,
            sector=bp.sector,
            state=dict(bp.base_state),
            behavioural_rules=dict(bp.behavioural_rules),
            reward_penalty=bp.reward_penalty,
        )
        for bp in selected_bps
    ]
    return agents, policy_signals


def simulate_economy(policy_text: str, time_steps: int = 8) -> Dict[str, object]:
    """Simulate the evolution of the economy over time for the top impacted actors."""

    agents, policy_signals = build_agents_for_policy(policy_text)
    timeline: List[Dict[str, StateVector]] = []

    for t in range(time_steps):
        snapshot = {}
        for agent in agents:
            snapshot[agent.name] = agent.step(policy_signals, time_index=t)
        timeline.append(snapshot)

    return {
        "policy_signals": policy_signals,
        "selected_actors": [agent.name for agent in agents],
        "timeline": timeline,
    }


if __name__ == "__main__":
    sample_policy = (
        "Increase transfer payments, ease small-business credit, and tighten emissions standards for utilities."
    )
    result = simulate_economy(sample_policy, time_steps=6)
    for idx, step in enumerate(result["timeline"], start=1):
        print(f"\nTime step {idx}")
        for actor, state in step.items():
            print(f"- {actor}: {state}")
