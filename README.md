# dsc180-D06
## AI and Human Contributions

We used an AI assistant (Claude, by Anthropic) only as an tool throughout this project to explain
the subject and referenced papers, unfamiliar code and concepts, draft code, suggest terminal commands, debug errors, and draft
written sections. All direction, execution, verification, and final decisions were ours.

**How the AI was used**
- **Explaining:** walking through the ClimateBench baseline notebooks, the JEM slab-ocean code,
  and concepts such as Q-fluxes, ERF, Gregory regression, and ECS.
- **Drafting code:** shared data utilities, pattern scaling and its JAX re-implementation, the
  NRMSE metric, the Q-flux construction script, and the Gregory, ECS, and warming-pattern analyses.
- **Debugging:** environment and package conflicts, file paths, grid-orientation errors, and
  plotting issues.
- **Writing:** first drafts of the report introduction, which we then revised.

**How we kept it accountable**
Every script was run by us, and every result was checked against an external reference or a
physical expectation, such as the published ClimateBench leaderboard, the model's own log
messages, and basic energy-balance reasoning. Several AI suggestions were wrong and caught this
way, including an incorrect NRMSE normalization, a dependency downgrade that broke our Python
environment, a Q-flux file with latitudes in reverse order, a Q-flux that ignored sea-ice
temperatures, and a premature conclusion that the Gregory regression "does not work" for our
setup, which we challenged and investigated further.

**Human contributions**
We chose the research questions, ran all simulations and notebooks, decided which results to
trust, and wrote the final interpretations. The slab-ocean control and abrupt-CO₂ runs analysed
here were produced by Coleman.
