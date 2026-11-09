# A Small Switch in a Vast Machine

## How Astra’s Qwen edit works—and what its geometry can teach us

{{FIG:cover}}

The first clue was **85 unfinished answers**.

When Astra asked Qwen3.5-0.8B to solve 96 practice problems, eleven answers were correct. The other 85 ran through the 32,768-token budget without reaching an answer. Their final passages repeated the same phrases over and over. Before any geometric analysis, Astra already had a useful lead: a recognizable pattern of failure.[^logs]

The eventual repair changed selected columns in two MLP matrices. Takehi and colleagues reported that MATH-500 accuracy rose from **7.9% to 48.1%**, while the fraction of answers hitting the token limit fell from **93.0% to 33.1%**. Gradient-based training was not allowed. We distinguish those reported results from our own experiments, which examine how the edit works inside the model and how far local geometry can explain its effects.[^source]

How did an LLM find where—and in which direction—to intervene among hundreds of millions of parameters?

A trained network has places where internal features can be measured and their effects changed. We can observe activations, modify the matrices that write them into the residual stream—the representation that successive layers read and update—and check what happens to the output. Astra used this structure to find a compact edit, with a different direction at each edited layer. During inference, that edit acts as **activation steering whose strength depends on the current state**.

**This essay connects a model-written correction to a mathematical framework for designing the next edit.** We reconstruct its action as activation steering, measure how well Fisher geometry predicts changes in the output distribution, and explain which corrections a shared weight matrix can express. Together, these give an engineer a way to move from a proposed correction to a candidate weight edit.

The practical sequence is to test a correction in activations, check whether the permitted weights can implement it across inputs, and measure the resulting behavior. Information geometry helps with the middle of that sequence: it describes reachable changes and gives small changes in output distribution a common measure. Our experiments also locate a limit—the local prediction breaks down at the full strength of Astra's edit.

We will follow **the failure Astra observed, the program it wrote into the weights, and the geometry that describes how that program acts**. The final section asks how this could become a repeatable method across models, and what would need to happen for a successful edit to help produce the next improvement.

<details class="proof">
<summary>Experiment inventory and scope</summary>

> **What the evidence covers.** The large benchmark improvement comes from the original authors. Our experiments comprise a reconstruction on 40 prefixes, local-KL measurements on 32 prefixes, and a steering-control experiment using artificially repetitive text. A prefix is the sequence of preceding tokens supplied to the model. A small generation pilot is complete; full MATH-500 reproduction and the fixed-program search study remain unfinished. The paintings are allegories. The mathematical surfaces come from explicitly defined models.

Astra did not need to understand the meaning of every weight. It found a measurable failure and a linear projection where a short rule could act on it. The edit reads a scalar from incoming activations and adds a correction along a fixed direction. This raises two questions: **can one shared matrix produce the corrections we want, and how much do those corrections change the output distribution?** The first concerns a reachable subspace; the second concerns a metric.

The division of labor is important. The logs provide evidence about how Astra discovered the failure and selected an edit. The algebra explains how the stored edit executes once it has been found. Fisher geometry supplies a local, prospective description of output change and reachability; it does not explain why Astra discovered this particular direction or why it chose these layers and gain.

</details>

## 1. What could Astra observe?

Start with the task and the tools. The initial prompt gave Astra Qwen's architecture, functions for reading and writing weights, labeled practice problems, and tools for collecting outputs and activations. It even named repetition as an example of a failure worth repairing. Astra was asked to classify failures, test interventions rather than stop at correlations, vary their strength, and check the results on unused questions.[^prompt]

The run with a budget of 50 evaluation-tool calls could repeat this cycle of observation and intervention. Here, “50” counts tool calls, not model forward passes: one call could run hundreds of generations or evaluate supplied text under teacher forcing, where supplied tokens rather than sampled tokens determine each next-token evaluation. Astra inspected answers, measured internal quantities, tried edits, and used behavior on held-out questions to decide what to try next.

The logs preserve failed requests and inconclusive early results. A single-neuron ablation and a subsequent removal of a direction in one layer each produced a noisy +6.25 percentage-point change on 32 problems. A later edit to layers 15 and 19 together produced a much larger held-out improvement. The repair emerged through this sequence of trials.

Three features of the setup made the search tractable. First, the failure was visible: many answers followed the same pattern of prolonged repetition ending at the token limit. Second, access to activations and log-probabilities made it possible to compare repetitive and non-repetitive text quantitatively. Third, the MLP down-projection provided a linear path for writing into the residual stream. This structure made a short corrective program possible. The results alone do not tell us how much of the success came from Astra's reasoning, the prompt, the tools, or a failure that happened to be amenable to this kind of repair.

The shorter 10-call and 25-call runs identified the same broad failure but produced less effective edits: a weaker rank-one-style operation across many output matrices, and an attention-head ablation. The same diagnosis did not guarantee the same solution.

The public logs report three contrasting runs. All three are the original authors' results, not our reruns.

| Reported run | Evaluation budget | Edit described in the released brief | Reported MATH-500 accuracy |
| --- | ---: | --- | ---: |
| p10c | 10 evaluation API calls | Remove a loop direction from 30 matrices across layers 8–22, gain 1.5 | 12.3% |
| p25c | 25 evaluation API calls | Disable head 7 in layer 15 | 14.1% |
| p50c | 50 evaluation API calls | Subtract a masked direction in 64 columns per layer, MLP layers 15 and 19, gain 6 | 48.1% |

The table compares the programs each run actually produced. Evaluation budget measures resources; it does not by itself establish why the scores differ. The original README also reports the unedited accuracy of 7.9% and the changes in truncation rate. The 48.1% result, like the other entries, is an original report rather than an independent full reproduction here.[^source]

It would also be premature to conclude that the edit gave Qwen new mathematical abilities. Although the evaluation used mathematics problems, the observed bottleneck was finishing an answer. On the original confirmation set, the edited model completed many answers that had previously exhausted the token budget. Correct answers after editing were longer, rather than merely ending sooner. That pattern is consistent with removing repetition that prevented existing competence from producing an answer. It does not establish that a low-rank intervention added general arithmetic or algebraic ability.

Evaluation should preserve that distinction. Alongside overall accuracy, we should count answers rescued by the edit, previously correct answers lost, completed but incorrect answers, answer lengths, and degeneration into repeated whitespace.

The practical lesson is to make the failure measurable before choosing an edit family. An aggregate score cannot tell us whether errors share an internal state, reflect a missing feature, arise from decoding, or have unrelated causes. The 85/96 observation narrowed the search before a geometric interpretation was needed.

## 2. How does a 0.016% edit act as a program?

{{FIG:anatomy}}

One edited matrix can move different inputs by different amounts—even in opposite directions. The figure shows that same rule executed in two ways: stored in the weights on the left, and applied to the activations on the right. Corresponding outputs coincide.

<details class="proof">
<summary>How to read the linear example</summary>

The figure uses a precisely defined toy model to compare two operations. It is not a recording of Qwen activations. Take three states, A=(1,0), B=(0,1), and C=(-1,0), with W=I, k=(1,0), and d=(0,.7). On the left, one edited matrix transforms an entire grid of inputs. On the right, the same correction is applied separately to each activation. Both are drawn in fixed output coordinates: hollow points mark original outputs and filled points mark edited ones. At full strength, A moves by +d, B stays still, and C moves by −d.

The animation gradually increases the edit strength in both panels. Corresponding points coincide throughout. This illustrates an algebraic identity, not Astra’s search or measured Qwen dynamics. A single external push on A would not, by itself, specify a policy for later inputs.

</details>

The actual edit acts on the MLP down-projections in layers 15 and 19. Let \(W\) map MLP-neuron activations \(a(x)\) into the residual stream. The mechanism follows from an exact identity:

The 0.016% figure is an approximate parameter-count ratio: 131,072 targeted entries (2×64×1024) out of roughly 800 million model parameters. It is not a relative weight norm or the fraction of the two MLPs.
\[
\boxed{\Delta W=d k^\top \quad\Longrightarrow\quad \Delta h(x)=d\,[k^\top a(x)].}
\]
Read a scalar from the current state, multiply a direction vector by it, and add the result to the activations. The scalar varies continuously and can be positive or negative. Calling it a “gate” does not make it a binary loop detector. Astra's idealized edit at each layer takes the form

\[
\Delta W=-g\,u\,(m\odot u^\top W),
\]

where \(u\) is a unit direction, \(m\) is a mask selecting columns, and \(g\) is the gain controlling intervention strength. The resulting activation change is

\[
\Delta h(x)=\Delta W a(x)=-g\,u\,[m\odot(u^\top W)]a(x).
\]

The vector \(u\) determines **which direction to move**. The scalar multiplying it depends on the current MLP activations and determines **when and how strongly to move**. The weights therefore store a state-dependent steering rule, rather than an instruction to add the same vector every time.

At a linear projection, the possible interventions can be organized as follows:

- Add a constant vector, also achievable with an editable bias: \(\Delta h=d\).
- Read one scalar and use it to scale a direction: \(\Delta h=d[k^\top a(x)]\).
- Assign an output vector to each neuron: \(\Delta h=\Delta W a(x)\).
- Specify an arbitrary change for each input: \(\Delta h=v(x)\).

This is a classification of interventions at a linear interface, not a theorem about minimizing KL. A bias-free matrix, for example, cannot necessarily add the same nonzero shift to every possible input; an editable bias makes that operation available. Nor can a single shared matrix generally implement an arbitrary field of changes. Its Jacobian determines which changes are reachable.

In the actual edit, each of the two layers has its own direction and 64 selected columns. The published formula remains rank one after masking, before rounding to bfloat16. The saved weight difference also contains small rounding residuals. The original evaluation used those saved weights; the ideal formula explains the mechanism. Keeping the two distinct matters for reconstruction.

In E1, we compared the saved weight edit with its additive activation intervention on fixed prefixes. Across 40 prefixes in FP32, the maximum logit difference was \(2.0\times10^{-5}\), and the maximum KL was \(9.6\times10^{-12}\). Differences were larger in bfloat16: a maximum logit difference of 0.3047 and maximum KL of 0.00713. This experiment checks numerical reconstruction of the edit. It does not provide a new task-performance result.

The gain matters to the interpretation. A gain of one removes the selected columns' component along \(u\). A gain of six changes that component to approximately negative five times its original value, reversing it rather than merely removing it. Where the selected neurons write along the contrast direction, the edit adds a contribution in the opposite direction. Calling this only an “ablation” obscures the feedback-like behavior. It still does not establish that the gate selectively detects every naturally occurring loop.

With edits at multiple layers, the activations used by each intervention matter too. Layer 19 receives inputs already affected by the edit at layer 15. The additive identity remains exact when \(a(x)\) comes from the edited execution. Replaying gate values recorded from the unedited model does not exactly reproduce the edited model. The switch must be evaluated from the activations that arrive at that moment.

<details class="proof">
<summary>When the weight edit and activation hook agree</summary>

For one linear projection, arrange the neuron activations for each sample in rows:

~~~python
# W: [residual_dim, neuron_dim]; u: unit direction
readout = mask * (u @ W)
D = -gain * u[:, None] * readout[None, :]
edited_write = a @ (W + D).T
hooked_write = a @ W.T + a @ D.T
~~~

The two expressions agree in exact arithmetic. To reconstruct the released model, use the actual saved difference \(D=W_{\mathrm{saved}}-W_{\mathrm{base}}\), including its rounding residuals. At each later edited layer, use activations that already reflect the earlier interventions.

</details>

{{FIG:reconstruction}}

We also completed a small generation pilot. On eight selected problems, we generated one sample per problem from the unedited model and four from the edited model. The respective correct counts were 0/8 and 8/32; truncation counts were 8/8 and 18/32. This pilot checked the generation pipeline and measured runtime. Because sampling counts differ between conditions, it is not a matched estimate of performance on the full benchmark.[^pilot]

{{INTERACTIVE:steering}}

## 3. Did the switch need to depend on the state?

{{FIG:controls}}

To test whether the state-dependent switch matters, replace it with a constant intervention along \(u\). If the two have the same effect on repetitive text, which better preserves text associated with useful output? The preregistered E2 comparison included Astra's original gate, a constant intervention, shuffled gate values, and a random control. Two limitations constrain what the measurements can tell us.

First, the repetition was constructed. The E2 manifest explicitly describes a test using synthetic continuations under teacher forcing, rather than repetition generated by the model. It uses 16 calibration cases and 32 evaluation cases. Repetitive text is made by taking a reference-answer prefix and repeating its final 16-token block. Qwen did not enter these loops during generation. The experiment probes the edit's response to a controlled state; it does not directly reproduce the failure Astra observed.

Second, the effects were not successfully matched. The constant, shuffled, and random conditions all failed the predeclared effect-matching check. Their calibration residuals relative to the target were +0.1312, −0.0266, and +0.1701, respectively. With effects unmatched on the tested gain grid, we cannot make a clean comparison of side effects at equal benefit.

The results are still informative about response patterns. The original gate responds differently from the controls, and we can compare responses at teacher-forced positions in productive and repetitive sequences. But this does not show that thermostat-like, state-dependent control is necessary during natural generation.

All three calibration curves bracket the target, so a finer gain search may improve the match. The next experiment should keep the same controls, refine that search, and construct contrast pairs from answers the model actually generates on practice problems. Calibration and evaluation data must remain separate, and matching the target effect must be a prerequisite for comparison. We can then ask how the controls change the probability of entering or escaping a loop during generation, beyond their effect on the log-probability of a synthetic continuation.

## 4. How far can Fisher geometry predict the effect?

{{FIG:fisher}}

A norm measured directly in weight coordinates can be a poor guide to an edit's effect. Two weight changes of the same size can change the output distribution by very different amounts. For a fixed prefix and a small activation displacement \(v\), the next-token KL has the local expansion

\[
\mathrm{KL}(p_h\Vert p_{h+v})
=\tfrac12 v^\top F_hv+o(\|v\|^2),
\]

where \(F_h\) is the output Fisher matrix pulled back to activation coordinates. This metric measures how distinguishable the distributions are. It does not measure whether an answer is correct or its meaning is preserved. A low-KL intervention can still damage an unmeasured capability. Keeping KL low on text associated with useful output is a design objective we choose, not a guarantee supplied by Fisher geometry.

E3 tested this prediction on 32 evaluation prefixes. We estimated Fisher separately for each prefix using finite differences, and fitted the scalar coefficients of the norm baselines on 16 separate calibration prefixes. The evaluation strengths—0.03, 0.1, 0.3, and 1.0—were excluded from the local fit.

At strength 0.03, the median absolute prediction error on a log scale was 0.013 decades for Fisher, compared with 0.479 for the weight-norm baseline. At strength 1, the errors were 0.462 and 0.420, respectively. By this measure, Fisher was substantially more accurate for the small intervention, but slightly worse at the original edit strength.

At full strength, the median ratio of the Fisher prediction to measured KL, computed separately for each prefix, was about **0.38**. This is the median of those ratios, not the ratio of the two medians. Agreement at the small strengths used to estimate or check derivatives cannot be counted as independent evidence of predictive accuracy.

Strength 1 corresponds to Astra's gain-six edit. At the strength that improved the original task, the local Fisher approximation was no longer accurate. Task improvement therefore does not validate the local approximation: an edit can leave the local regime and still help. The metric describes nearby changes; it does not automatically describe an entire generation.

These measurements also condition on a fixed prefix. During generation, changing the next token changes subsequent prefixes and their Fisher matrices. A theory of the whole sequence would need to specify both the sequence distribution and the intervention path. The fixed-prefix result cannot simply be extended to the entire rollout. Our theory note keeps these questions separate and leaves guarantees over finite trajectories as an open extension.

<details class="proof">
<summary>Conditions of the local-KL experiment</summary>

We computed KL in FP32 using ordinary softmax distributions at fixed prefixes. The calculation excludes the top-k and top-p truncation used during generation. The primary finite-difference step was 0.001; steps of 0.003 and 0.01 were used to check the derivative estimate. Only one saved edit direction was tested.

The activation-norm baseline uses activations from the intervened execution, so it is not a prediction available from the unedited state alone. This experiment also does not establish that a Fisher estimate transfers to another direction or another prefix.

</details>

## 5. What can one matrix do—and what changes with the metric?

{{FIG:reachability}}

The figure puts the requested and realizable changes on one coordinate chart. At each of eight inputs, a blue arrow shows the requested change and a violet arrow the change produced by the shared matrix. All eight violet arrows depend on the same matrix, so they cannot be adjusted independently. Gold segments join the two endpoints: they are the part of each request that the matrix does not realize.

<details class="proof">
<summary>Eight-input toy setup and least-squares diagnostics</summary>

Put the eight inputs in the columns of \(A\) and their desired activation shifts in the columns of \(V\). One 2×2 matrix update must fit all of them. If we measure error with the Euclidean metric \(F=I\), the least-squares solution is

\[
\Delta W^*=\arg\min_{\Delta W}\|\Delta W A-V\|_F^2=VA^+.
\]

The animation varies the update from zero to this solution. All eight arrows move together, while the gold segments connect their endpoints to the requested endpoints. Even at the optimum, the residual's Frobenius norm is 65.6% of the norm of the desired field.

That number comes from this constructed example, not from Qwen. The animation interpolates toward a solution we have already computed; it does not record Astra's search or the steps of an optimizer.

The model file contains the inputs, requested changes, solution, and residual. The normal-equation residual \(\|(V-\Delta W^*A)A^\top\|_F\) is about \(1.5\times10^{-16}\). The remaining error is nonzero, but orthogonal to the space of realizable changes. What happens to the closest solution if we change the metric?

</details>

### Which change is closest?

{{FIG:duality}}

The right panel shows the desired activation change and the line of changes that the permitted weight edits can realize. The ellipses measure mismatch from the desired change. Shrinking an ellipse until it touches the line identifies the closest reachable change in the Fisher metric; the open gray point marks the different Euclidean projection. On the left, a natural-gradient step in weight space maps to that same reachable change. The synchronized arrows show the two routes to one result.

The operational question is simple: if several inputs request different changes, how much of that field can one shared matrix realize, and which realizable change is least costly? The figure makes the first constraint visible; the formal conditions are in the foldout.

The weight-to-activation relation is an **adjoint and metric relationship**. Let \(E\) map the permitted weight coordinates \(\theta\) to the state at the intervention interface, with derivative \(J=DE\). This state may collect activations across many inputs. With the downstream computation held fixed, it must capture every effect of the weight edit on that computation. This is the requirement for a complete intervention interface.

Pulling the activation-space metric \(F\) back through \(J\) gives the weight-space metric \(G_\theta=J^\top FJ\).

A natural gradient gives a direction for changing an objective, with distance measured by the chosen metric. We label that metric below the gradient symbol. The equation says that two ways of choosing a feasible activation change agree:

In identifiable coordinates, or on the quotient by null directions described below, a smooth objective \(C\) satisfies
\[
\boxed{J\,\nabla_{G_\theta}(C\circ E)
=\Pi_{\operatorname{im}J}^{F}\,\nabla_{F}C.}
\]
The left-hand side is the activation change induced by moving the weights along their natural gradient. The right-hand side projects the activation-space natural gradient onto the changes those weights can realize, using the Fisher metric. **Locally, taking a natural-gradient step in weight space and mapping it to activations gives the same result as taking the natural gradient in activation space and projecting it onto the reachable directions.**

<details class="proof">
<summary>Metric-gradient definition and the shared-input metric</summary>

To specify which metric defines the gradient, we use the **metric-gradient operator \(\nabla_M\)**. It converts the differential of a scalar objective into a direction using the metric \(M\). For positive-definite \(M\) and every tangent vector \(v\), it satisfies
\[
\langle\nabla_M f,v\rangle_M=df[v]
\quad\Longleftrightarrow\quad
\nabla_M f=M^{-1}\partial f.
\]
Here \(df[v]\) is the first-order change in \(f\) along \(v\), and \(\partial f\) collects the ordinary coordinate derivatives in a column. Setting \(M=I\) recovers the Euclidean gradient; using a Fisher metric gives the **natural gradient**. The subscript specifies the metric, not the direction of differentiation. This operation raises the differential's index with the inverse metric. It is distinct from taking the covariant derivative of a vector field.

For a single linear write projection, \(J_x=a(x)^\top\otimes I\), and averaging the pullback metric over inputs gives \(\mathbb E[(aa^\top)\otimes F_x]\). Every input in that average uses the same parameter update. Steering one input therefore cannot be chosen independently of the effects on the others.

</details>

<details class="proof">
<summary>The projection identity: proof and conditions</summary>

In differential-geometric index notation,
\[
(\nabla_M f)^i=M^{ij}\partial_j f,
\]
where \(M^{ij}\) denotes the inverse metric and repeated indices are summed. The covariant derivative of a scalar satisfies \(\nabla_i f=\partial_i f\); raising its index gives the gradient vector. Our subscript \(M\) is a metric label, whereas the index \(i\) in \(\nabla_i\) denotes a coordinate direction.

Assume \(F\) is positive definite on identifiable activation coordinates. Write \(b=\partial C\) for the coordinate representation of the differential \(dC\). If \(G_\theta\) is singular, interpret its gradient on the quotient by \(\ker J\). Choosing the Moore–Penrose representative, which minimizes Euclidean norm in the chosen weight coordinates, gives
\[
J\,\nabla_{G_\theta}(C\circ E)
=J(J^\top FJ)^+J^\top b.
\]
The \(F\)-orthogonal projector is \(J(J^\top FJ)^+J^\top F\). Applying it to \(\nabla_F C=F^{-1}b\) gives the same expression.

If the Fisher metric itself is singular, work on an identifiable quotient space and require the objective differential to vanish on null directions. Simply substituting a Euclidean pseudoinverse does not make the statement coordinate-invariant.

The downstream map must remain fixed. For interventions at multiple sites, account for their dependencies through a complete interface or an explicit intervention graph. The identity establishes local agreement; it does not imply that optimization trajectories coincide after finite steps.

</details>

<details class="proof">
<summary>Amari preview and vector-covector distinction</summary>

The second is **Amari's duality between natural and expectation coordinates**, which appears when the output is represented by an exponential family. If \(h\) denotes softmax natural coordinates and \(\eta=\nabla\psi(h)\) the expected sufficient statistics, then

\[
\dot\eta=q
\quad\Longleftrightarrow\quad
\dot h=F(h)^{-1}q.
\]

A straight path in expectation coordinates generally becomes a curve in natural coordinates, and vice versa. This provides a mathematical basis for “dual steering” at the output representation. It does not make the weights themselves expectation coordinates, or make every hidden layer dually flat. Nonlinear intermediate computation, rank deficiency, and unidentifiable directions each require their own conditions.

Together, these relations give us a mathematical account of the Qwen edit: a rank-one update can apply a direction with input-dependent strength, and a Fisher-weighted alternative could reduce output KL while preserving a fixed local target effect. They do not explain how Astra chose its contrast examples, why the two layers looked promising, or why it tried gain six. The logs show a search that adapted to observations, not evidence that these choices came from a Fisher calculation.

It is important here to distinguish a vector from a covector. An activation difference between two classes of text is a **displacement vector**: it describes how the representation changes. The differential of a behavioral objective is a **covector**: it measures how much that objective changes under a small displacement. Using the Fisher metric to relate them requires specifying the metric and a local output model.

Replacing every contrast vector with \(F^{-1}v\) does not automatically produce natural-gradient steering; it confuses two different mathematical objects. The proposed E5 experiment specifies the objective, the positions at which cost is measured, damping, and the effect-matching procedure in advance. Its candidate direction is defined under those conditions, rather than by a general prescription to precondition every useful direction.

The phrase “compiled into weights” carries the same constraints. It means that an editable matrix implements a particular field of activation changes across a set of inputs. The resulting program need not be independent of those inputs, cannot generally implement arbitrary fields, and is not guaranteed to preserve every other function. A shared matrix links its effects across contexts. Even a successful edit is one choice from a constrained family of functions, and its side effects are part of what that program does.

</details>

For edit design, this suggests an order of work. First check which requested changes the permitted weights can produce. Then compare feasible changes using a cost defined on the outputs we want to preserve. A metric can help choose among them; whether the resulting edit helps during generation still needs a separate test.

### Can we get the same effect at lower cost?

{{FIG:efficiency}}

Every point on the gold line in this figure produces **the same target effect to first order**. The ellipses show a quadratic cost for changing the output distribution we want to protect. Moving along the line toward the smallest ellipse that touches it preserves the target effect while reducing that cost. The blue point is the least costly change. The paired bars make the comparison explicit: the effect stays fixed while the cost falls.

For a positive-definite metric \(F\), nonzero objective differential \(b\), and target effect \(b^\top d=c\), the optimal displacement is

\[
 d_* = \frac{c\,F^{-1}b}{b^\top F^{-1}b}.
\]

The figure uses an explicitly defined two-dimensional quadratic model. The points represent equal-effect candidates, not measurements of Astra's edit. The animation moves among those candidates to explain the geometry; it does not replay the search.

To use this idea in a real editing experiment, we would need to estimate the metric, match the target effect, and evaluate behavior on held-out data. Information geometry provides a candidate direction and a way to compare it with alternatives. That role does not imply that Astra's earlier search was a natural-gradient algorithm.

<details class="proof">
<summary>Minimum cost at a fixed target effect</summary>

Minimize \(\tfrac12 d^\top Fd\) subject to \(b^\top d=c\). The stationarity condition gives \(Fd=\lambda b\), and substituting the constraint yields the solution above. Write any other feasible point as \(d=d_*+r\). Then \(b^\top r=0\), so

\[
\tfrac12d^\top Fd
=\tfrac12d_*^\top Fd_*+\tfrac12r^\top Fr.
\]

This result is exact for the quadratic form. Interpreting it as a KL cost requires a local approximation. The metric can be defined in activation coordinates or in the permitted weight coordinates, related by the pullback \(G_\theta=J^\top FJ\). The rank and null-direction conditions above still apply.

</details>

## 6. Why do shared weights make some steering requests impossible?

{{FIG:impossibility}}

Consider a model with just one weight, \(h_w(x)=wx\), evaluated at \(x=1\) and \(x=-1\). Changing the weight by \(\delta w\) shifts the two activations by \(+\delta w\) and \(-\delta w\). The shifts necessarily point in opposite directions.

Now suppose we want the same positive shift \(c\) for both inputs. Direct activation steering can add \((c,c)\). A shared weight edit would have to satisfy both \(\delta w=c\) and \(-\delta w=c\), which is impossible for \(c\ne0\).

This small example exposes the constraint. A weight edit cannot specify an independent intervention for every input: the same parameter change links all of its effects. More generally, consider the local linearization, in which \(J\) maps permitted weight displacements to activation displacements. Within this linearized model, only changes in \(\operatorname{im}J\) are reachable. For a reachable change \(u\), with a positive-definite metric \(M\) measuring resource cost in weight space, the minimum cost is

\[
\min_{Jv=u}\tfrac12v^\top Mv
=\tfrac12u^\top(JM^{-1}J^\top)^+u.
\]

If the requested change lies outside \(\operatorname{im}J\), no edit satisfies the constraint, and the cost is \(+\infty\). Evaluating the pseudoinverse expression alone must not be used to assign a finite cost to an unreachable request. These feasibility and cost statements are exact for a linear interface. For a nonlinear model, an infeasible linearized request does not by itself rule out a finite edit. Dual variables can certify infeasibility in the linearized model. Such a certificate can guide a practical decision: add a bias, edit another layer, relax a constraint on a protected context, or stop searching.

Fisher alone also cannot select a sparse edit. Different weight changes can induce the same local activation field and therefore have the same output Fisher cost. Choosing 64 columns, two layers, or a particular sparse mask requires an additional resource constraint, architectural prior, or search rule. Astra's mask was observed as part of a successful program; the metric does not determine it uniquely.

## 7. Where does Amari's duality enter?

We can now visualize the other duality: the relationship between natural and expectation coordinates.

Take a three-class softmax and remove the redundancy by fixing the third logit at zero. The remaining two logits are natural coordinates \(h=(h_1,h_2)\). Define
\[
\psi(h)=\log(1+e^{h_1}+e^{h_2}),\qquad
\eta=\nabla\psi(h)=(p_1,p_2).
\]
Expectation coordinates give the probabilities themselves; natural coordinates give the log-odds \(h_i=\log(p_i/p_3)\). These are different coordinates for the same distribution. Since the Hessian of \(\psi\) is the Fisher matrix,
\[
\boxed{\dot\eta=q\quad\Longleftrightarrow\quad\dot h=F(h)^{-1}q.}
\]
Moving at constant velocity in expectation coordinates generally requires the velocity in natural coordinates to vary. The animation shows the same evolving distribution in both coordinate systems.

{{FIG:coordinates}}

The coordinate transformation is computed analytically; it is not a landscape measured inside Qwen. One view uses log-odds, while the other uses the probability simplex. The synchronized bars make it clear that the two paths, despite their different shapes, represent the same probabilities. On the interior of the simplex, the dual potential is negative entropy, \(\varphi(\eta)=\sum_i p_i\log p_i\).

This duality must be distinguished from the weight-to-activation projection identity. **Weights and hidden activations are not automatically each other's Legendre-dual coordinates.** An arbitrary intermediate layer need not form a minimal exponential family or a dually flat space.

We can pull the output metric back to a hidden layer. That alone does not give the layer a globally dually flat structure.

Even so, once we specify a behavioral objective, the geometry suggests a concrete edit to test. Let the covector \(b\) measure the desired first-order change, and let \(F_{\mathrm{prod}}\) measure cost on contexts whose useful outputs we want to preserve. A damped candidate direction is proportional to \((F_{\mathrm{prod}}+\lambda I)^{-1}b\). Whether it helps must be tested at a matched target effect. This remains a research proposal; it has not been shown to improve on Astra's edit.

## 8. What have we learned, and what remains open?

For an engineer deciding what to do next, the evidence supports the following workflow. These are deductions and proposed tests from this case, not validated product guarantees.

| Decision | Evidence-led action | What it establishes |
| --- | --- | --- |
| Diagnose a recurring failure | Measure completion, repetition, truncation, and internal contrasts before editing weights. | Whether a shared, measurable bottleneck is a plausible target. |
| Prototype an activation hook | Implement the candidate correction at the MLP down-projection and inspect its state-dependent response. | Whether the proposed rank-one rule can be reproduced numerically. |
| Match controls | Tune constant, shuffled, and random controls to the same target effect before comparing side effects. | Whether a claimed benefit survives equal-effect comparison. |
| Sweep intervention strength | Measure local and larger strengths separately, including the full-dose setting. | Where a local metric is predictive and where it stops being so. |
| Test shared constraints | Fit requested changes across inputs and inspect residuals and reachability. | Whether one shared weight matrix can express the proposed steering field. |
| Validate held-out generation | Evaluate naturally generated, held-out trajectories and task outcomes with matched sampling. | Whether a fixed-prefix or teacher-forced result transfers to generation. |

**Astra found a small program that runs inside Qwen.** The edit reads the current activations and changes how the model writes into its residual stream. Logs record the failure, contrast measurements, causal interventions, gain sweeps, and improvements on questions not used in the search.

The contribution of this analysis is to connect that executable rule to a geometry of possible edits. The exact linear identity tells us how to reproduce a weight edit with an activation intervention. The shared-weight constraint tells us whether different inputs can receive the corrections we want. The Fisher metric gives us a local cost in output KL. These are useful parts of an editing procedure: test the correction, check that it can be stored in the permitted weights, and compare its effects on outputs we want to preserve.

Our E1 experiment reconstructed the edit as an additive intervention and confirmed numerical agreement in FP32. E3 found that Fisher geometry predicted small KL changes at fixed prefixes well, but the prediction broke down at the original gain-six strength. That result locates the limit of the approximation; the original edit still improved the reported task results. E2 measured responses to constructed sequences. Because those sequences did not arise from natural generation and the control effects were not matched, it cannot establish whether the switch is necessary during generation.

### From a working correction to a reusable method

The mathematics applies wherever an editable linear map writes into a model's activations. Directions, gates and useful intervention sites still have to be found in each model, but we can ask the same engineering questions at each site. This is the basis for a research program: **find a useful steering rule, establish which part can be implemented by shared weights, and validate the stored rule on new inputs.** Independent [WSP follow-up experiments](https://arxiv.org/html/2611.09382v1) already report gains on several tasks, alongside uneven transfer and searches that retained the original model. Reliable discovery is a concrete problem to solve.

This connects directly to current work on model control. Anthropic's [Assistant Axis](https://www.anthropic.com/research/assistant-axis) uses activation constraints to stabilize behavior. Goodfire's [manifold steering](https://www.goodfire.com/research/manifold-steering) studies interventions that follow the geometry of representations. Our complementary question is which of those useful interventions a shared weight edit could implement. The projection identity gives that question a precise local form. Some requested corrections lie outside the reachable space; identifying them helps us decide when to change the edit family or keep an intervention in activation space.

The next step is a preregistered comparison with the original behavioral target held fixed. It should use naturally generated trajectories as well as synthetic controls, keep discovery data separate from evaluation data, and test whether a direction informed by the metric predicts fewer side effects before committing to expensive generation runs. Negative controls should preserve intervention magnitude while disrupting the proposed mechanism. If the comparison succeeds, geometry becomes a tool for designing edits; if it fails, the logs still show that an LLM can write a program into a large network when it can observe a failure, measure internal contrasts, access a suitable write path, and iterate within a sufficient budget.

### What would make the improvement recursive?

A useful edit can matter twice: it can improve the task we measured, and it might improve the model's ability to help make subsequent edits. The second effect is the one to test for recursive self-improvement. Removing an obstacle to an existing ability could help, even if the edit adds no new ability.

A concrete experiment would let an edited model help propose or evaluate the next set of edits, then compare it with the unedited model doing the same work. Give both the same tools, feedback and compute budget. After each accepted edit, re-estimate the activations and local metric before designing another. In the recursive arm, each accepted edited checkpoint becomes the editor for the next round.

Evaluate the resulting edits independently on held-out tasks, count failed searches as well as successes, and repeat over successive rounds. This tests whether the improvement strengthens the process that produces further improvements. The Astra–Qwen case establishes no such feedback loop yet; this is the research program that the mechanism makes concrete.

That distinction matters in the wider discussion of AI research automation. [OpenAI's account of research acceleration](https://openai.com/index/research-acceleration-view-inside-openai/) and [METR's assessment at Anthropic](https://metr.org/blog/2026-11-05-anthropic-ai-rd-acceleration-assessment/) examine how AI contributes to further model development. WSP supplies a particularly inspectable candidate operation within that process: a model proposes a weight change, we can examine what it does, and independent evaluation decides whether to retain it. Our proposal is to use the geometry to help design and test those operations, then ask whether the models become better at the process itself.

<details class="proof">
<summary>Scope, unresolved questions and prior work</summary>

Here, “one model and one failure mode” refers to the Astra–Qwen case analyzed in this essay. It does not describe the full range of experiments in the [original WSP report](https://rikiyat.github.io/blog/weight-space-programming/), and this case alone cannot establish how WSP scales in general. There is no evidence that Astra used Fisher geometry. Nor have we shown that gated rank-one instructions are generally easy to discover. We have not established whether the edit added an ability or removed an obstacle to an ability the model already had.

Information geometry guides experiments and helps identify impossible requests; it does not replace diagnosis or behavioral validation. The generality of the linear identity gives us a common mathematical framework across compatible model interfaces. Whether useful edits can be found reliably across model families, and whether improvements persist over successive rounds, are empirical questions.

These connections build on prior work: Amari's natural gradient, [Steer2Edit](https://lilywenglab.github.io/Steer2Edit/), and [FishBack](https://arxiv.org/html/2605.17231v2). This essay tests their scope in one WSP case; it does not claim the dualities are new. Its contribution is the connection between the released program, numerical reconstruction, local geometry and a procedure for testing future edits.

</details>

**Astra's edit gives us a rule we can inspect, execute and test. The proposed path is to turn that understanding into a repeatable method across models, then ask whether an improved model becomes better at making the next edit.** This is where model control meets recursive self-improvement: the correction becomes a candidate step in a process that models could help improve.

### Sources

[^source]: [Original Weight-Space Programming report and README](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/README.md), with the published 7.9%, 48.1%, 93.0%, and 33.1% values mirrored in [`blog/data/measurements.json`](data/measurements.json).
[^prompt]: [Released Astra initial prompt](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/turns/t001/prompt.txt).

Additional primary/local references: [Astra final report](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/work_notes/final_report.md); [Japanese version of this essay](../ja/); [activation information-geometry note](repro/activation_information_geometry.md); [E5 v2 plan](repro/e5_dual_edit_v2.md).


[^logs]: [Astra's final report](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/work_notes/final_report.md), including the initial diagnosis and held-out practice confirmation.
[^pilot]: [Completed pilot summary](data/pilot_status.json); sampling counts differ between arms.

*The allegorical paintings were generated in Figma AI and assembled as editable figures in Figma Design. Their use of watercolor, layered translucent surfaces, and explanatory annotations draws inspiration from [BetaTomorrow / deep Manifold](https://x.com/BetaTomorrow). These illustrations were made for this essay; they were neither created nor endorsed by that account.*

