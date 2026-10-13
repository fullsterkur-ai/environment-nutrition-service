# Context-Aware Nutrition & Hydration Timing Engine: Methodology & Research Framework

**Project:** FullStrength.ai  
**Component:** Environmental Quality & Nutrition Timing Optimizer (TEQS)  
**Student ID:** IT23166660  
**Status:** `EXPERIMENTAL_RULE_BASED` / `EDUCATIONAL_PROTOTYPE_UNVALIDATED`

---

## 1. Overview & Research Objective

The **Context-Aware Nutrition & Hydration Timing Engine** (Stage 3) is an educational research prototype within the TEQS service. Its objective is to provide transparent, situational nutrition and hydration guidance for resistance training athletes based on:
1. **Workout Context:** Planned duration, planned intensity, and exercise modality.
2. **Environmental Conditions:** Ambient temperature, relative humidity, and facility setting (`INDOOR_AC`, `INDOOR_NON_AC`, `SEMI_OUTDOOR`, `OUTDOOR`).
3. **Athlete Readiness:** Prior sleep duration and subjective readiness rating.
4. **Digestive Proximity:** Time interval elapsed since the last substantial meal.
5. **Exploratory TEQS Baseline:** The exploratory environmental quality score (labeled strictly `EXPERIMENTAL_UNVALIDATED`).

---

## 2. Non-Prescriptive Design Philosophy

Sports nutrition recommendations in academic literature emphasize that fluid and carbohydrate requirements vary widely depending on body mass, sweat rate, heat acclimation, metabolic efficiency, and gastrointestinal tolerance.

To prevent pseudoscientific overprecision, this engine adheres to the following constraints:
* **No Fabricated Fluid Volumes:** The engine does **not** prescribe arbitrary volumes (e.g., "drink 500 ml every 15 minutes"). Instead, it reinforces baseline hydration, accessibility, and drinking to thirst and comfort.
* **No Exact Gram Prescriptions:** It does **not** mandate exact macronutrient quantities (e.g., "consume 35g carbohydrates and 25g protein"). It highlights qualitative meal composition principles.
* **No Supplement Endorsements:** It does **not** prescribe creatine, beta-alanine, caffeine, or proprietary pre-workout formulas.
* **Educational Scope:** All recommendations are educational concepts intended for athlete awareness and research evaluation, not clinical or dietetic prescriptions.

---

## 3. Rule Catalog & Situational Rationales

| Rule Identifier | Trigger Condition | Educational & Physiological Rationale |
|---|---|---|
| `RULE_ENV_HEAT_HUMIDITY` | ($T \ge 28^\circ\text{C}$ or $\text{RH} \ge 70\%$) and environment is non-AC | High ambient temperature and humidity impair evaporative heat dissipation, increasing sweat rate, thermal strain, and cardiovascular drift. Hydration accessibility and cooling between sets are prioritized. |
| `RULE_ENV_INDOOR_AC` | `environment_type == INDOOR_AC` | Air conditioning stabilizes environmental heat strain. However, cool ambient air can blunt thirst sensation while respiratory and baseline fluid losses continue. |
| `RULE_ENV_OUTDOOR_DISCLAIMER` | Source is `OPENWEATHER` or environment is outdoor/semi-outdoor | Regional weather observations represent outdoor atmospheric estimates and may diverge significantly from indoor facility microclimates. |
| `RULE_DURATION_SHORT` | $\text{Duration} \le 45\text{ min}$ | Brief resistance training relies primarily on resting muscular glycogen and phosphagen systems. Plain water access for comfort is adequate; intra-workout carbohydrate supplementation is not required. |
| `RULE_DURATION_EXTENDED` | $\text{Duration} \ge 75\text{ min}$ | Extended training progressively depletes glycogen stores and accumulates sweat losses. Educational guidance highlights access to easily digestible carbohydrates and deliberate fluid pacing. |
| `RULE_INTENSITY_HIGH` | $\text{Intensity} \ge 8/10$ | High relative intensity elevates anaerobic glycolysis, metabolic heat production, and rate of perceived exertion (RPE). Carbohydrate availability before and after exercise supports workload capacity. |
| `RULE_MEAL_FASTED_OR_REMOTE` | Last meal was fasted or $>4\text{ hours}$ ago | Circulating blood glucose and hepatic glycogen stores may be lower. An optional light, easily digestible carbohydrate snack 30–60 minutes prior helps support training stamina if tolerated. |
| `RULE_MEAL_RECENT` | Last meal was $<1\text{ hour}$ ago | Active gastric digestion diverts blood flow to the splanchnic circulation. Heavy food or large fluid volumes immediately before high-intensity lifting can cause gastrointestinal fullness or reflux during abdominal bracing. |
| `RULE_RECOVERY_DEPRECATED` | Sleep $<6\text{ hours}$ or readiness $\le 2/5$ | Reduced sleep and subjective readiness elevate session RPE. Athletes are encouraged to prioritize hydration and post-workout restorative nutrition rather than relying on heavy stimulants. |
| `RULE_TEQS_CONSTRAINED` | $\text{TEQS Score} < 60/100$ | Cumulative environmental and readiness limiters are active. Perceived exertion will likely be higher. (Strictly labeled as an unvalidated exploratory indicator, not a medical threshold). |

---

## 4. Phase-Specific Guidance Framework

1. **Pre-Workout Phase:**
   * Objective: Fluid readiness and gastrointestinal comfort.
   * Focus: Establishing baseline hydration in the hours prior; adapting pre-session fueling to meal timing proximity; avoiding gastric fullness before heavy lifting.
2. **During-Workout Phase:**
   * Objective: Maintaining hydration accessibility, cooling comfort, and stamina.
   * Focus: Drinking to thirst; adapting fluid access to AC vs non-AC settings; considering intra-workout carbohydrate availability only for extended sessions (>75 min).
3. **Post-Workout Phase:**
   * Objective: Rehydration and restorative nutrition.
   * Focus: Rehydrating gradually over subsequent hours; consuming a balanced meal containing protein and carbohydrates according to regular daily meal patterns; including dietary electrolytes following warm/humid training.

---

## 5. Assumptions & Known Limitations

1. **Uncalibrated TEQS Baseline:** The TEQS score is an exploratory scaffold (`ml/teqs_baseline.py`) with uncalibrated weights. It must never be used as a medical safety cutoff.
2. **Outdoor Weather Discrepancy:** Outdoor OpenWeather readings cannot capture gym-specific air conditioning, commercial fans, or indoor thermal pockets.
3. **Subjective Self-Reporting:** Pre-workout readiness (1–5) and meal proximity are self-reported and subject to recall bias.
4. **Individual Sweat Rate Variability:** Individual sweat rates vary significantly (0.5 to 2.5+ L/hour); generalized guidance cannot replace individualized sweat rate testing.

---

## 6. Expert Review Requirements

Before advancing toward clinical validation or athletic deployment, the following areas require formal review by registered sports dietitians (RD / CSSD) and exercise physiologists:
* **Meal Timing Windows:** Validation of the 30–60 min pre-workout snack recommendation across varying digestive tolerances.
* **Intra-Workout Carbohydrate Thresholds:** Evaluating whether 75 minutes is the appropriate duration threshold for strength-specific versus endurance-specific training.
* **Electrolyte Guidance:** Establishing clear guidelines on when whole-food dietary electrolytes suffice versus when electrolyte beverages are advantageous in tropical training environments.

---

## 7. Future Real-User Evaluation Protocol

To transition from an educational prototype to an empirically evaluated system:
1. **Usability & Comprehension Study:** Present timing recommendations to resistance training athletes ($N \ge 30$) to evaluate comprehension, clarity, and absence of restrictive dietary behavior triggers.
2. **Ecological Momentary Assessment (EMA):** Correlate adherence to timing guidance with real-world session RPE, perceived gastrointestinal comfort, and session completion rates.
3. **Physiological Grounding:** In future phases, integrate empirical sweat rate estimations and real gym ambient sensor data to replace outdoor approximations.
