# Event Sequencing Module for a Model of Early Social Cognition

**Python implementation of a sequence-learning and planning model using self-organizing maps (SOMs)**  
*MEi:CogSci mobility project, 2021–2022*

## Project overview

How might a computational system learn to recognise events as sequences, anticipate what happens next, and select a sequence associated with a desired outcome?

This project explores that question by reimplementing the **C-block model**, a connectionist model of sequence learning and planning. The work was motivated by research on **event segmentation**: the way people organise continuous experience into meaningful events.

The implementation was developed in **Python** and progressed from a simple self-organizing map trained on animal attributes to a C-block architecture with two SOMs:

- **Sequencer SOM:** learns representations of sequences and predicts the next element from what has already been observed.
- **Planner SOM:** associates learned sequences with result and reward values, allowing the system to retrieve a sequence (or *plan*) associated with a target outcome.

The experiments use small illustrative datasets—animal attributes and sequences of letters forming names—to make the network's behaviour easier to inspect. This is a research prototype, not a general-purpose language model or a validated model of human social cognition.

## How does a self-organizing map work?

A **self-organizing map (SOM)** is an unsupervised neural network that maps multidimensional inputs onto a usually two-dimensional grid of neurons. Each neuron has a vector of weights with the same dimensionality as the input.

During training, the network:

1. **Initialises** the neurons' weight vectors.
2. **Receives an input vector** from the training data.
3. **Finds the best-matching unit (BMU)** by comparing the input with neuron weights using Euclidean distance.
4. **Updates the BMU and neighbouring neurons** so their weights become more similar to the input.
5. **Repeats** the process, adjusting learning parameters according to the training procedure.

Because nearby neurons learn related representations, similar inputs tend to activate nearby regions of the map. Importantly, the network learns these similarities **without predefined class labels as training targets**.

### Example 1: Learning similarities between animals

To understand the SOM algorithm before implementing the C-block, we trained a **20 × 20 SOM (400 neurons)** on 16 animals, each described by **13 binary attributes** such as size, number of legs, feathers, hunting, flying and swimming.

For example, a *duck* was represented by attributes including `small = 1`, `2 legs = 1`, `feathers = 1` and `swim = 1`. Animals with similar attribute vectors tended to occupy related regions of the trained map:

- **Birds:** duck, goose, dove, hen, owl, hawk and eagle.
- **Hoofed animals:** horse, zebra and cow.
- **Other mammals:** wolf, dog, fox, tiger, lion and cat.

The map also revealed an instructive limitation: **owl and hawk** had identical attribute vectors in this dataset, as did **horse and zebra**, so the network could not distinguish either pair from those features alone. This illustrates that what a neural network can learn depends on the information encoded in its inputs.

## How does the C-block model work?

The C-block combines two SOMs that represent different aspects of an event sequence.

```text
                     C-BLOCK MODEL

       +--------------------+    +--------------------+
       |   PLANNER SOM      |    |   SEQUENCER SOM    |
       |                    |    |                    |
       |  Result            |    |  Tonic             |
       |  Reward            |    |  Context           |
       |  Tonic             |--->|  Recent            |
       +--------------------+    |  Next              |
                                 +--------------------+

       Retrieves a plan          Learns and predicts
       for a desired outcome     elements in a sequence
```

### The sequencer: predicting what comes next

The sequencer learns four linked representations:

| Component | Meaning |
| --- | --- |
| **Tonic** | A representation of the overall sequence. |
| **Context** | Information accumulated from preceding elements. |
| **Recent** | The immediately preceding element. |
| **Next** | The element the network learns to predict. |

In the project, **individual letters were sequence elements**, while **complete names were sequences**. The network learned, for instance, that the letters in `TANJA.` follow a particular order, where `.` marks the end of the sequence.

A simplified illustration of the input construction is:

| Sequence representation as it develops | Previous context | Recent | Next |
| --- | --- | --- | --- |
| `T` | — | — | `T` |
| `TA` | — | `T` | `A` |
| `TAN` | `T` | `A` | `N` |
| `TANJ` | `TA` | `N` | `J` |
| `TANJA` | `TAN` | `J` | `A` |
| `TANJA.` | `TANJ` | `A` | `.` |

*This table shows the readable letter-level structure from the project report. In the actual implementation, the components were encoded numerically, with decay factors used to represent the position and recency of elements. In the final training representation, the completed tonic was used consistently across the elements of a given sequence.*

The implemented sequencer used a **50 × 50 SOM (2,500 neurons)**. Each neuron had **400 weights**: 100 each for tonic, context, recent and next. During prediction, the network was queried with the available information to infer the next element.

The project compared an initial version trained on a predefined dataset with a later version that **learned incrementally as each new sequence element was presented**. The incremental implementation also introduced an exploration mechanism for selecting an alternative neuron when the closest match was insufficient.

### The planner: selecting a learned sequence

The planner SOM learns associations between:

- **Result:** an encoded outcome associated with a sequence.
- **Reward:** a value assigned to that sequence.
- **Tonic:** the sequence representation learned by the sequencer.

Unlike the sequencer, which learns element by element, the planner is trained **at sequence boundaries**, after a complete sequence has been processed. The implemented planner used a **30 × 30 SOM (900 neurons)** with **201 weights per neuron**: 100 for tonic, 100 for result and one for reward.

When supplied with result and reward information, the planner can retrieve a likely tonic representation. The sequencer then uses that representation to reconstruct the corresponding sequence, one element at a time.

## Example 2: When a prediction changes

One of the most interesting tests allowed a user to enter letters manually while the trained model predicted the next letter and inferred the most likely sequence.

Consider the overlapping names **`martin`** and **`marusa`**. Both begin with `mar`, so early input does not uniquely identify the intended name.

In one test using the network's **hard output** (the winning neuron's weights), the interaction proceeded as follows:

| User enters | Model predicts next |
| --- | --- |
| `m` | `a` |
| `a` | `r` |
| `r` | `t` |
| `u` | `s` |
| `s` | `a` |
| `a` | `.` (end of sequence) |

After `mar`, the network initially favoured the plan for **`martin`** and predicted `t`. When the user instead supplied `u`, the input contradicted that prediction. The implementation **reduced the attention weight assigned to the previously inferred tonic** (by a factor of 0.6), allowing the inferred plan to shift toward **`marusa`**.

The report records **4 correct predictions out of 5 evaluated next-letter predictions (80%)** in this particular interactive example, excluding the end-of-sequence marker. This is a result for a single demonstration, **not an overall generalisation accuracy score**.

The project also explored **soft output**, which combines neuron weights according to their activity rather than selecting only the winning neuron. This made it possible to observe competing candidate sequences being active at the same time.

## Example 3: Exploring alternative plans

The final extension implemented **inhibition of return (IOR)** during planner testing. Instead of repeatedly selecting the same best-matching plan, the procedure temporarily inhibited a previously selected plan and its neighbourhood so that subsequent searches could retrieve alternatives.

In the reported test, the procedure returned **13 different learned sequences across 13 iterations**. This demonstrates alternative retrieval within the small training set; it does not establish general-purpose planning ability.

## Results and limitations

Across the reported experiments, the C-block implementations successfully **reconstructed all 13 names in the training dataset** under the documented test settings. The animal SOM produced broadly interpretable groupings based on the encoded attributes.

These findings should be interpreted within the scope of the prototype:

- **Small training set:** the C-block experiments used only 13 names.
- **No held-out generalisation test:** the model was not evaluated on previously unseen names or other sequence types.
- **Input representation matters:** identical feature vectors cannot be distinguished, as shown in the animal experiment.
- **Parameter sensitivity:** some reconstructions required adjusting attention weights, decay constants, exploration and output settings.
- **Prototype implementation:** the project produced Python scripts rather than a user interface.

Possible extensions include testing unseen sequences, expanding the dataset, generalising the input encoding beyond letters, and evaluating how reliably the model adapts when predictions are contradicted.

## Project documentation

The implementation was developed incrementally. The accompanying documentation files follow the same progression:

| Documentation file | What it covers | Status noted in original README |
| --- | --- | --- |
| `read_me_animals.md` | Introductory SOM trained on animal attributes; separate from C-block. | Available with preview code |
| `read_me_cblock_with_sequencer1.md` | First sequencer implementation trained using a predefined dataset. | Available with preview code |
| `read_me_cblock_with_sequencer2.md` | Incremental sequencer learning after each new element. | Unavailable |
| `read_me_whole_cblock.md` | Combined sequencer and planner SOMs. | Unavailable |
| `read_me_whole_cblock_withIOR.md` | Planner testing with inhibition of return; builds on the whole C-block implementation. | Unavailable |

Each detailed README describes its corresponding script and functions, followed by credits and references. The first four describe stages that can be understood independently; the IOR documentation depends on the whole-C-block version.

## Background and references

Selected references from the project report:

- Asan, U., & Ercan, S. (2012). *An Introduction to Self-Organizing Maps*. https://doi.org/10.2991/978-94-91216-77-0_14
- Ritter, H., & Kohonen, T. (1989). *Self-Organizing Semantic Maps*. https://doi.org/10.1007/BF00203171
- Takáč, M., Knott, A., & Sagar, M. (2020). *SOM-Based System for Sequence Chunking and Planning*. https://doi.org/10.1007/978-3-030-61609-0_53
- Zacks, J. M., Speer, N. K., Swallow, K. M., Braver, T. S., & Reynolds, J. R. (2007). *Event perception: A mind-brain perspective*. https://doi.org/10.1037/0033-2909.133.2.273
