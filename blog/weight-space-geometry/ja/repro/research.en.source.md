# A Small Switch in a Vast Machine

## Why Astra could edit Qwen—and what information geometry actually explains

{{FIG:cover}}

The first clue was **85 unfinished answers**.

Astra had asked Qwen3.5-0.8B to work through 96 practice problems. Eleven answers were correct. The other 85 reached a 32,768-token ceiling. Their tails were full of repeated phrases. Before inspecting a geometric metric, the editor had found something unusually useful: a failure with a recognizable shape.[^logs]

The repair eventually changed selected columns in two MLP matrices. In the original authors' benchmark, MATH-500 accuracy rose from **7.9% to 48.1%**, and the fraction of answers reaching the limit fell from **93.0% to 33.1%**. No gradient-based training was allowed. These are the results reported by Takehi and colleagues; our experiments below test the edit's execution and local geometry.[^source]

How can an LLM find its way through hundreds of millions of parameters?

A trained network exposes structured interfaces: features that can be measured, matrices that write them into a residual stream, and outputs that can be checked. Astra used these interfaces to discover a compact intervention, with a separate direction at each edited layer. When the model runs, that intervention executes as a **state-dependent steering command**.

Information geometry enters at the next step. It gives us a ruler for the command's effect and a language for the commands that shared weights can express. It also makes a prediction we can test—and, at the full edit strength, watch fail.

The short answer is that Astra did not need to assign a meaning to every weight. It found a measurable failure and a linear interface where a compact rule could act on that failure. The edit reads a scalar from each arriving activation and writes a correction along a direction. Geometry then asks two separate questions: **can one shared writer produce the requested corrections, and how much do those corrections change the output distribution?** The first is a question about a reachable subspace; the second is a question about a metric. Neither requires treating raw parameter count as the size of an unstructured search.

This essay follows three objects: **the failure Astra observed, the program it wrote, and the geometry through which that program acts**.

> **Reading the evidence.** The large benchmark improvement is the original authors' result. Our measurements are a 40-prefix reconstruction, a 32-prefix local-KL experiment, and a synthetic steering-control probe. A small generation pilot has completed; full MATH-500 reproduction and the fixed-program search study remain unfinished. Paintings are allegories. Curved mathematical surfaces are explicitly defined models.

## 1. What did Astra actually have to solve?

The first question is not “what did Astra know about information geometry?” It is “what decision problem did the environment present?” The initial prompt exposed Qwen’s architecture, functions for reading and writing weights, labeled practice problems, and tools for running the model and collecting outputs and activations. It also gave repetition as an example of a failure worth repairing. Astra was asked to classify failures, intervene rather than rely on correlation, vary intervention strength, and check on unused questions.[^prompt]

The 50-call run therefore had a measurement loop. Here “50” counts evaluation-tool calls: a call could contain hundreds of generations or teacher-forced texts. It does not mean 50 individual model forward passes. It generated or evaluated practice cases, inspected outputs, measured internal quantities, wrote candidate edits, and used held-out behavior to decide what to try next. The logs record failed requests and inconclusive early interventions as well as the final edit. A single-neuron ablation and a subsequent single-layer directional removal each gave a noisy +6.25 percentage-point change on 32 problems. Later, a joint edit at layers 15 and 19 produced a much larger held-out improvement. The path from diagnosis to edit was not a one-shot flash of insight.

The environment made this tractable in three ways. First, the failure was visible in output behavior: long repeated tails and systematic truncation. Second, the model could be queried for activations and log-probabilities, so “repetition” could be turned into a measurable contrast. Third, the architecture exposed a linear write path in the MLP down-projection. These facts explain why a short program was available. They do not isolate how much came from Astra’s reasoning, the prompt, the tools, or the favorable structure of the failure.

The shorter 10-call and 25-call runs are important counterevidence to a simplistic story. They identified the same broad failure but wrote weaker programs: the 10-call run modified many output matrices with a gentler rank-one-style operation, while the 25-call run disabled an attention head. Same diagnosis does not imply the same solution. A useful account must explain both the success and the nearby failures.

The public log records three contrasting runs. These are reported runs, not our reruns, and the search budget is a resource count rather than a causal explanation of the score changes:

| reported run | search budget | edit described in the released brief | reported MATH-500 |
| --- | ---: | --- | ---: |
| p10c | 10 evaluation API calls | remove a loop direction from 30 matrices across layers 8–22, gain 1.5 | 12.3% |
| p25c | 25 evaluation API calls | switch off head 7 in layer 15 | 14.1% |
| p50c | 50 evaluation API calls | masked direction subtraction in 64 columns per layer, MLP layers 15 and 19, gain 6 | 48.1% |

The table compares the programs the agent actually produced, not evidence that more search time alone caused the final result. The primary README reports the corresponding 7.9% unedited baseline and the truncation changes; the 48.1% row is the original report, not an independent full reproduction here.[^source]

This is also why the headline should not be “Astra learned mathematics by editing weights.” The edit was evaluated on mathematics, but the measured bottleneck was completion. On the original confirmation set, the edited model rescued many answers that had previously exhausted the token budget; correct edited answers were longer rather than simply terminating early. That pattern is compatible with removing a recurrent obstacle that prevented existing competence from reaching an answer. It is not evidence that a low-rank intervention injected a general arithmetic or algebraic ability. The distinction is operational: count rescues, losses, completed-but-wrong answers, answer length, and whitespace degeneration, rather than treating accuracy as a single undifferentiated capability score.

For ML engineers, the practical lesson is to instrument the failure before choosing an edit family. A model-wide score can tell you that something is wrong, but it does not tell you whether the failure is a shared state, a missing feature, a decoding artifact, or a collection of unrelated errors. The 85/96 observation narrowed the search space before any geometric interpretation was needed.

## 2. How can a 0.016% edit behave like a program?

{{FIG:anatomy}}

The mechanism plate is an exact paired toy, not a recording of Qwen activations. It uses three declared states A=(1,0), B=(0,1), and C=(-1,0), with W=I, k=(1,0), and d=(0,.7). The installed writer ΔW=d kᵀ and the hook d(kᵀa) produce the same local change on the current input: +d for A, zero for B, and −d for C. A one-shot external push on A does not automatically become a policy for later states. The causal animation is therefore a visualization of the identity and its contrast, not evidence that a natural Qwen loop was detected.

The final edit touched the MLP down-projections in layers 15 and 19. Let \(W\) map MLP-neuron activations \(a(x)\) into the residual stream. The program appears in one exact identity:
\[
\boxed{\Delta W=d k^\top \quad\Longrightarrow\quad \Delta h(x)=d\,[k^\top a(x)].}
\]
Read a scalar from the current state, multiply a direction by it, and write the result. The scalar is signed and continuous; a “gate” need not be a binary detector of loops. Astra's idealized change at each layer has the form

\[
\Delta W=-g\,u\,(m\odot u^\top W),
\]

where \(u\) is a unit direction, \(m\) selects columns, and \(g\) is the gain. Applying the changed matrix gives

\[
\Delta h(x)=\Delta W a(x)=-g\,u\,[m\odot(u^\top W)]a(x).
\]

The vector \(u\) says **which way** to move the residual stream. The bracketed scalar says **when and how strongly** to move it. It depends on the current MLP activations. This is therefore not a constant steering vector. It is a state-dependent steering rule stored in the weights.

A useful analogy is a river and a rudder. A weight edit changes the shared rule that maps the current state to the next push; direct steering changes the immediate push at one moment. Both can be state-dependent—the rudder can respond to the current flow—but a shared weight edit must use the same rule across every input. The river is an intuition for a state-dependent policy, not a literal neural dynamical landscape.

For a linear writer, the expressivity ladder is explicit:

- constant activation vector (or editable bias): \(\Delta h=d\);
- one scalar readout: \(\Delta h=d[k^\top a(x)]\);
- one output vector per neuron: \(\Delta h=\Delta W a(x)\);
- arbitrary field: \(\Delta h=v(x)\).

These are intervention families at a direct linear interface, not a minimum-KL theorem. A nonzero constant shift is not automatically writable by a bias-free matrix on arbitrary inputs; an editable bias would make it available. The last field is generally not realizable by one shared matrix; reachability is constrained by the image of its Jacobian.

The two layers have their own directions. The released formula selected 64 output columns per layer. Before bfloat16 rounding, it is rank one after masking; the stored difference also contains small rounding residuals. The exact implementation distinction matters: the saved edited weights are the object evaluated by the original run, while the ideal formula is the object used to reason about the program. In the project’s E1 reconstruction, comparing the stored edit with an additive intervention on fixed prefixes gives a maximum logit difference of \(2.0\times10^{-5}\) and maximum KL \(9.6\times10^{-12}\) in FP32 (40 prefixes). The corresponding bfloat16 comparison is much less exact, with maximum logit difference 0.3047 and maximum KL 0.00713. That is numerical evidence about reconstruction, not a new behavioral evaluation.

The gain also changes the interpretation. A gain of one would remove the selected columns’ component along \(u\). Gain six changes that component to approximately negative five times its original value: a reversal, not a deletion. Calling it “ablation” hides the feedback-like behavior. Where the selected neurons write toward the contrast direction, the edit adds an opposing contribution. This does not establish that the gate selectively detects every natural loop.

There is one subtle dynamic point. With two edited layers, layer 19 receives an input already affected by layer 15. The additive identity remains exact when \(a(x)\) is read in the edited run, but replaying switch values recorded on the unedited trajectory is not an exact simulation of the edited model. This is why “a switch” should mean a live activation-dependent gate, not a frozen list of coefficients.

<details class="proof">
<summary>The patch and its activation-hook equivalent</summary>

For row-batched neuron activations and one linear writer:

~~~python
# W: [residual_dim, neuron_dim]; u: unit direction
readout = mask * (u @ W)
D = -gain * u[:, None] * readout[None, :]
edited_write = a @ (W + D).T
hooked_write = a @ W.T + a @ D.T
~~~

The expressions agree in exact arithmetic. To reconstruct the released model, use its actual saved difference \(D=W_{\mathrm{saved}}-W_{\mathrm{base}}\), including rounding residuals. At a later edited layer, use the activations that arrive after earlier interventions.

</details>

{{FIG:reconstruction}}

The completed generation pilot is separate. Across eight selected problems, the base arm used one sample per problem and the edited arm used four. They produced 0/8 and 8/32 correct samples, with 8/8 and 18/32 truncations. This unequal-sampling pilot exercised the generation path and measured runtime; it is not a matched full-benchmark estimate.[^pilot]

{{INTERACTIVE:steering}}

## 3. Was the switch really doing the work?

{{FIG:controls}}

This is the right causal question: would a constant push along \(u\), matched to the same effect on repetitive text, preserve productive text as well as the live switch? The registered E2 comparison was designed around Astra’s exact switch, a constant direction, shuffled switch values, and a random control. But the implemented measurement has a limitation that changes what it can support.

The current E2 manifest labels the scope “synthetic teacher-forced continuation; repetitive states are constructed, not model generated.” It uses 16 calibration cases and 32 evaluation cases. The repetitive target is made by taking a reference-answer prefix and repeating its final 16-token block; it is not a rollout in which Qwen naturally entered a loop. This is a controlled probe of the edit’s response to a constructed state, not a direct replication of the generation-time failure that motivated Astra.

There is a second limitation: the calibration curves for the constant, shuffled, and random variants all failed the predeclared exact effect-matching check. The constant arm’s residual was +0.1312, the shuffle arm’s −0.0266, and the random arm’s +0.1701 in the recorded calibration. When target effects have not been matched on the tested gain grid, a clean “same benefit, less collateral damage” comparison is unavailable. The rows are still useful diagnostics: the exact switch has a different response pattern from the controls, and productive and repetitive teacher-forced positions can be compared. But the result should not be promoted to “the thermostat is necessary in natural generation.”

The recorded calibration curves bracket the target in all three arms; a finer gain search could therefore improve matching. The smallest decisive follow-up is to refine the gain search and preserve the same controls while constructing contrast pairs from model-generated practice trajectories, with separate calibration and evaluation sets, and to report effect matching as a prerequisite rather than a footnote. Generation-level tests would then ask whether the control changes the probability of entering or escaping a loop, not merely the log-probability of a synthetic continuation.

## 4. What does Fisher geometry predict—and where does it fail?

{{FIG:fisher}}

A raw parameter norm is a poor ruler for a neural network. Two equal-sized weight changes can produce very different output distributions. For a fixed prefix and a small activation displacement \(v\), the next-token KL has the local form

\[
\mathrm{KL}(p_h\Vert p_{h+v})
=\tfrac12 v^\top F_hv+o(\|v\|^2),
\]

where \(F_h\) is the output Fisher matrix pulled back to the activation coordinates. The metric measures distinguishability of distributions, not correctness or semantic preservation. A direction can have low KL while damaging an unmeasured capability; preserving productive KL is a chosen engineering objective, not a theorem of Fisher geometry.

E3 tests this local prediction on 32 evaluation prefixes. Each evaluation prefix has its own finite-difference Fisher estimate; 16 separate calibration prefixes fit the scalar coefficients of the norm baselines. Doses 0.03, 0.1, 0.3, and 1.0 are held out from the local fit. At dose 0.03, Fisher's median absolute log-prediction error is 0.013 decades, compared with 0.479 for the weight-norm baseline. At dose 1, those errors are 0.462 and 0.420. Fisher is much better locally and slightly worse by this summary statistic at the full dose.

The median of the per-prefix Fisher-to-measured KL ratios at full dose is about **0.38**. This is a median of ratios, not a ratio of medians. Small doses used to estimate or diagnose the derivative are not independent demonstrations of prediction.

Dose 1 is Astra’s gain-6 edit. That makes the result especially informative: the winning intervention is not evidence that the local Fisher model remains accurate at the winning strength. It is evidence that the edit can leave the local regime while still improving the original task. The metric describes a neighborhood; it does not automatically describe the full generated trajectory.

The evaluation is also next-token geometry on fixed prefixes. During generation, an intervention changes the next token, which changes later prefixes and therefore later Fisher matrices. A theory of the entire sequence would need to define the sequence distribution and the intervention path, not silently extrapolate a fixed-prefix result. The project’s theory note makes this distinction explicit and treats finite-trajectory guarantees as an unresolved extension.

<details class="proof">
<summary>What the local-KL experiment measures</summary>

These are fixed-prefix, ordinary-softmax KL calculations in FP32, excluding the top-k and top-p sampling truncations used during generation. The primary finite-difference step is 0.001, with 0.003 and 0.01 as derivative diagnostics. Only one saved edit direction is tested.

The activation-norm baseline uses activations on the intervened trajectory; it is not a prediction available solely from the original state. The experiment does not establish transfer of a Fisher estimate to a new direction or another prefix.

</details>

## 5. What can one writer realize—and how does the metric choose?

{{FIG:reachability}}

Read the two sheets as two views of the same eight inputs. The upper arrows show the requested moves. The lower arrows are tied to one shared matrix: they cannot be adjusted independently. The gold gaps lie within the lower sheet and show what that matrix cannot write. The vertical separation of the sheets is only a visual aid. This makes the shared-weight constraint visible before we change the ruler. Its eight inputs are the columns of \(A\); the eight desired activation shifts are the columns of \(V\). A single two-by-two update must fit them together. With Euclidean cost \(F=I\), the least-squares solution is

\[
\Delta W^*=\arg\min_{\Delta W}\|\Delta W A-V\|_F^2=VA^+.
\]

The animation scales this one matrix from zero to its solution, so all eight realized arrows change together. Gold segments connect each realized endpoint to its requested endpoint. At the optimum, 65.6% of the desired field's Frobenius norm remains in the residual. This is a property of the explicitly constructed toy field, not a Qwen measurement. The animation is an interpolation toward the solved matrix, not a recording of Astra's search or an optimizer.

The model file records the inputs, requests, solution, and residual. The normal-equation check \(\|(V-\Delta W^*A)A^\top\|_F\) is about \(1.5\times10^{-16}\): the residual is orthogonal to the realizable field family even though it is not zero. In the next figure, a non-Euclidean metric changes which realizable field is closest.

### Which geometry defines the closest move?

{{FIG:duality}}

The bowl is the cost of missing a desired move. Its bottom is the request itself. The blue curve contains the moves that the permitted weights can actually produce. The best feasible move is the lowest point on that curve; a different ruler changes where that point lies. The open gray marker shows the Euclidean answer for comparison.

The word “dual” hides two related but distinct constructions.

The first is an **adjoint and metric relationship** between weight edits and activation fields. Let \(E\) map permitted weight coordinates \(\theta\) into an intervention state, and let \(J=DE\). The state can stack activations across many inputs. It must be a complete interface for a fixed downstream computation.

Pulling an activation metric \(F\) back through \(J\) gives \(G_\theta=J^\top FJ\).

First, a notation that makes the ruler visible. **\(\nabla_M\) is the metric-gradient operator:** it turns the differential of a scalar objective into a direction using the metric \(M\). For positive-definite \(M\) and every tangent vector \(v\),
\[
\langle\nabla_M f,v\rangle_M=df[v]
\quad\Longleftrightarrow\quad
\nabla_M f=M^{-1}\partial f.
\]
Here \(df[v]\) is the first-order change of \(f\) along \(v\), and \(\partial f\) is the column of ordinary coordinate derivatives. With \(M=I\), this is the familiar Euclidean gradient; with a Fisher metric, it is the **natural gradient**. The subscript names the metric, not a differentiation direction. This is the operation of raising the differential's index with the inverse metric, rather than the covariant derivative of a vector field.

Using identifiable coordinates—or quotienting out null directions as described below—the central identity for a smooth objective \(C\) becomes
\[
\boxed{J\,\nabla_{G_\theta}(C\circ E)
=\Pi_{\operatorname{im}J}^{F}\,\nabla_{F}C.}
\]
Read the left side: take a natural-gradient weight step and observe the activation change it produces. Read the right side: take the desired natural steering and project it onto the changes those weights can realize. The two agree locally. The projection uses the Fisher ruler.

For one writer, \(J_x=a(x)^\top\otimes I\), and its expected pullback metric is \(\mathbb E[(aa^\top)\otimes F_x]\). The same parameter update must work across the inputs in that expectation. Shared weights couple the steering requests.

<details class="proof">
<summary>The projection identity: proof and conditions</summary>

In differential-geometric index notation,
\[
(\nabla_M f)^i=M^{ij}\partial_j f,
\]
where \(M^{ij}\) denotes the inverse metric and repeated indices are summed. The covariant derivative of a scalar satisfies \(\nabla_i f=\partial_i f\); raising its index gives the gradient vector. Our subscript \(M\) is a metric label, whereas the index \(i\) in \(\nabla_i\) denotes a coordinate direction.

Assume \(F\) is positive definite on identifiable activation coordinates, and let \(b=\partial C\) be the coordinate column representing the differential \(dC\). If \(G_\theta\) is singular, interpret its gradient on the quotient by \(\ker J\). The Moore–Penrose representative, which has minimum Euclidean norm in the chosen weight coordinates, gives
\[
J\,\nabla_{G_\theta}(C\circ E)
=J(J^\top FJ)^+J^\top b.
\]
The \(F\)-orthogonal projector is \(J(J^\top FJ)^+J^\top F\). Apply it to \(\nabla_F C=F^{-1}b\).

For singular Fisher metrics, work on identifiable quotient spaces and require the objective differential to annihilate null directions. Arbitrary Euclidean pseudoinverse substitutions are not, by themselves, a coordinate-invariant theorem.

The downstream map must be fixed. For multi-site interventions, include cross-dependencies through a complete interface or an explicit intervention graph. The identity is local, not a finite-step equivalence of optimized trajectories.

</details>

The second is **Amari’s natural/expectation-coordinate duality** inside an exponential-family output representation. If \(h\) is a softmax natural coordinate and \(\eta=\nabla\psi(h)\) is the expected sufficient statistic, then

\[
\dot\eta=q
\quad\Longleftrightarrow\quad
\dot h=F(h)^{-1}q.
\]

Straight motion in expected coordinates curves in natural coordinates, and vice versa. This is the mathematical basis for “dual steering” at the output representation. It is not the claim that raw weights are expectation coordinates or that every hidden layer is dually flat. Nonlinear intermediate computation, rank deficiency, and unidentifiable directions require separate conditions.

These dualities are useful lenses for the Qwen edit. They explain why a rank-one writer can implement a direction whose strength depends on the input, and why a Fisher-weighted alternative could reduce output KL for a fixed local behavioral effect. They do not explain how Astra selected the contrast examples, why those two layers were promising, or why gain six was tried. Those choices are visible in the logs as an adaptive search process, not as a Fisher computation.

The distinction between a vector and a covector is easy to lose in an explanatory article. A contrastive activation difference is a displacement vector: it says which way the representation moved between two classes of text. A behavioral objective supplies a differential, or covector: it says how the objective changes under a small displacement. Fisher converts between these types only after a metric and a local output model have been specified. Replacing every contrast vector with \(F^{-1}v\) would therefore be a category error, not an automatic upgrade to “natural” steering. The proposed E5 direction is meaningful because it defines its objective, cost positions, damping, and matching protocol in advance—not because every useful direction should be preconditioned.

This is also where the phrase “compiled into weights” needs care. Compilation means that a permitted writer matrix realizes a particular activation field on a set of inputs. It does not mean that the compiled program is independent of those inputs, that it can implement arbitrary fields, or that it preserves all other functions. The shared matrix couples contexts. A successful edit is a point in a constrained function family, and its side effects are part of the program’s semantics.

### Can the same behavioral effect cost less?

{{FIG:efficiency}}

The gold curve now has a different meaning: every point gives the **same first-order target effect**. Height measures the quadratic cost of changing the protected output distribution. Moving down that curve keeps the target effect while reducing this cost. The blue point is the cheapest such move.

For a positive-definite metric \(F\), an objective differential \(b\), and a target \(b^\top d=c\), that point is

\[
 d_* = \frac{c\,F^{-1}b}{b^\top F^{-1}b}.
\]

The figure uses an explicitly defined two-dimensional quadratic model. Its points are not measurements of Astra's patch, and the animation is a walk through equal-effect candidates, not Astra's search history. In a real editing experiment, one must estimate the metric, match the target effect, and evaluate held-out behavior. This gives information geometry a concrete role: it proposes a direction and a comparison to test. It does not retroactively turn Astra's successful search into a natural-gradient algorithm.

<details class="proof">
<summary>The equal-effect minimum</summary>

Minimize \(\tfrac12 d^\top Fd\) subject to \(b^\top d=c\). Stationarity gives \(Fd=\lambda b\); enforcing the constraint yields the expression above. If another feasible point is \(d=d_*+r\), then \(b^\top r=0\) and

\[
\tfrac12d^\top Fd
=\tfrac12d_*^\top Fd_*+\tfrac12r^\top Fr.
\]

This is an exact quadratic statement. Its interpretation as KL cost is a local approximation. The relevant metric may live in activation coordinates or in permitted weight coordinates; those are connected by the pullback \(G_\theta=J^\top FJ\), with the rank and null-direction qualifications already stated.

</details>

## 6. Why can’t one shared edit implement every steering request?

{{FIG:impossibility}}

Consider the smallest counterexample. A shared scalar weight produces \(h_w(x)=wx\). For inputs \(x=1\) and \(x=-1\), a weight change \(\delta w\) produces activation shifts \(+\delta w\) and \(-\delta w\). Suppose the desired direct intervention is the same positive shift \(c\) for both inputs. Direct activation steering can add \((c,c)\), but a single shared weight would require both \(\delta w=c\) and \(-\delta w=c\), which is impossible when \(c\ne0\).

The example is deliberately tiny because it exposes the engineering constraint. A weight edit is not an arbitrary per-input steering table. It is a field tied together by shared parameters. More generally, if \(J\) maps permitted weight coordinates to activation changes, only vectors in \(\operatorname{im}J\) are reachable. A minimum-cost edit for a reachable \(u\), with positive-definite resource metric \(M\), is

\[
\min_{Jv=u}\tfrac12v^\top Mv
=\tfrac12u^\top(JM^{-1}J^\top)^+u.
\]

Outside \(\operatorname{im}J\), the constrained problem is infeasible and its cost is \(+\infty\); the pseudoinverse expression must not be used alone to assign a finite cost. The dual variables can certify infeasibility when the requested steering lies outside the image. This is more than a metaphor: it tells an editor when to add a bias, edit another layer, relax a protected context, or stop searching.

It also explains why a sparse edit is not selected by Fisher alone. Multiple weight perturbations can produce the same local activation field, and all can have the same output Fisher cost. Choosing 64 columns, two layers, or a particular sparse mask requires an independent resource constraint, architecture prior, or search rule. The observed Astra mask is evidence about the successful program, not a unique consequence of the metric.

## 7. Where does Amari's duality enter?

There is a second duality, and it deserves its own picture.

For a minimal three-class softmax, choose natural coordinates \(h=(h_1,h_2)\), fixing the third logit at zero. Define
\[
\psi(h)=\log(1+e^{h_1}+e^{h_2}),\qquad
\eta=\nabla\psi(h)=(p_1,p_2).
\]
Expectation coordinates describe probabilities. Natural coordinates describe log-odds: \(h_i=\log(p_i/p_3)\). They describe the same distribution. The Hessian of \(\psi\) is the Fisher matrix, so
\[
\boxed{\dot\eta=q\quad\Longleftrightarrow\quad\dot h=F(h)^{-1}q.}
\]
A constant velocity in expectation coordinates generally requires a changing velocity in natural coordinates. The animation follows one probability path simultaneously in both systems.

{{FIG:coordinates}}

These are exact analytic coordinate maps, not measured Qwen landscapes. The natural coordinates are log-odds and the expectation coordinates lie on the probability simplex; the synchronized bars show that the two paths refer to the same probabilities. The dual potential is negative entropy, \(\varphi(\eta)=\sum_i p_i\log p_i\), on the simplex interior.

This natural/expectation duality is distinct from the weight-to-activation projection identity. **Weights and raw hidden activations are not automatically each other's Legendre-dual coordinates.** An arbitrary intermediate layer need not be a minimal exponential family or a dually flat space.

We can pull the output metric back into a hidden layer without claiming that all its coordinates inherit global dual-flat structure.

Once a behavioral objective is specified, geometry suggests a concrete candidate edit. If a covector \(b\) measures desired first-order change and \(F_{\mathrm{prod}}\) measures cost on protected productive contexts, a damped direction is proportional to \((F_{\mathrm{prod}}+\lambda I)^{-1}b\). Its value must be tested at matched target effect. It is a research proposal, not a demonstrated improvement to Astra's patch.

## 8. What should we conclude now?

The strongest supported conclusion is modest and useful. In this Qwen repair, Astra found a compact weight-space program that behaves like state-dependent activation steering. The released logs show the observable diagnosis, the contrast measurements, the causal probes, the gain sweep, and the held-out behavioral improvement. The project’s E1 measurement reconstructs the edit’s additive action with FP32 numerical agreement. E3 shows that Fisher geometry predicts small fixed-prefix KL changes well and fails at the full gain-6 dose. E2 provides a controlled synthetic probe but cannot establish necessity in natural generation because its repetitive states are constructed and effect matching failed.

Information geometry contributes three engineering ideas. First, output KL defines a distributional local cost; our test shows where its quadratic predictor improves on a norm baseline. Second, the permitted weight path determines which activation fields are reachable and how contexts are coupled. Third, natural and expectation coordinates provide a principled way to distinguish a direct steering direction from a metric-aware direction. These ideas can guide new experiments and diagnose impossible requests.

What remains unproven is just as important. We have no evidence that Astra used Fisher geometry. We have not shown that a rank-one guarded command is generally easy to discover. We have not established that this edit adds an ability rather than removes an obstacle that masked behavior the model already possessed. And one model, one failure mode, and one successful edit do not establish general WSP scaling.

The result is nevertheless a useful template for studying agentic model editing. Log the agent’s observations, not only its final patch. Preserve unsuccessful probes and the order in which hypotheses changed. Separate a reconstruction check from a behavioral replication. When introducing a theory after the fact, state which quantities it predicts before looking at the corresponding held-out measurements. Finally, design negative controls that preserve magnitude while breaking the proposed mechanism: shuffled gates, orthogonal directions, constant pushes, and shared-input counterexamples all make “it worked” harder to explain away.

For an engineer deciding what to build next, this suggests a small stack of tools rather than a single grand theory. A diagnosis layer finds repeated or otherwise structured failures. A contrast layer converts the behavior into measurable internal differences. A reachability layer asks whether the allowed weights can express the requested intervention across contexts. A cost layer estimates output-side collateral damage. A validation layer checks natural generation and held-out tasks. Astra’s run demonstrates one successful traversal of that stack; the geometric account supplies candidate abstractions for the reachability and cost layers. It does not replace the diagnosis or validation layers.

The decisive next step is a preregistered comparison that keeps the original behavioral target fixed, uses natural generated trajectories as well as synthetic controls, separates discovery from evaluation, and tests whether a metric-aware direction predicts a cleaner edit before the expensive generation run. If that experiment succeeds, geometry becomes a design tool. If it fails, the logs still teach us something durable: an LLM can program a large network when the environment exposes a visible failure, a measurable contrast, a reachable write path, and enough budget to iterate.

These connections have substantial prior work. Amari's natural gradient is established mathematics; [Steer2Edit](https://lilywenglab.github.io/Steer2Edit/) converts steering vectors into component-level rank-one edits, and [FishBack](https://arxiv.org/html/2605.17231v2) explicitly studies a pulled-back Fisher metric for activation steering. This essay reads a particular WSP case and tests its limits; it does not claim that the underlying dualities are new.

**A trained model can expose small programmable interfaces inside a vast learned computation.** The research task is to establish when those interfaces are discoverable, which commands they can express, and how far a local ruler can guide us before measurement must take over.

### Sources

[^source]: [Original Weight-Space Programming report and README](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/README.md), with the published 7.9%, 48.1%, 93.0%, and 33.1% values mirrored in [`blog/data/measurements.json`](data/measurements.json).
[^prompt]: [Released Astra initial prompt](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/turns/t001/prompt.txt).

Additional primary/local references: [Astra final report](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/work_notes/final_report.md); [project Japanese article](../index.html); [activation information-geometry note](repro/activation_information_geometry.md); [E5 v2 plan](repro/e5_dual_edit_v2.md).


[^logs]: [Astra's final report](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/work_notes/final_report.md), including the initial diagnosis and held-out practice confirmation.
[^pilot]: [Completed pilot summary](data/pilot_status.json); sampling counts differ between arms.

*Original allegorical paintings were generated in Figma AI and composed as editable Figma Design plates. Visual inspiration: the scientific watercolor, layered transparency, and explanatory annotation of [BetaTomorrow / deep Manifold](https://x.com/BetaTomorrow). These are original illustrations; no endorsement or authorship by that account is implied.*

The selected cover is an original Figma-composed pale graphite/watercolor allegory with two adult male Kabuki actors in slight spring/dance motion, a near-white timber canal, boats, and a preserved serpentine river layout; the figures and river are visual intuition, not historical reproduction or measured Qwen evidence.
