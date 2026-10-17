#!/usr/bin/env python3
"""Generate synchronized English/Chinese student and solution notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def source(text: str) -> list[str]:
    text = dedent(text).strip("\n") + "\n"
    return text.splitlines(keepends=True)


def markdown(text: str, cell_id: str) -> dict:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": source(text)}


def code(text: str, cell_id: str, tags: list[str] | None = None) -> dict:
    metadata = {"tags": tags} if tags else {}
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": metadata,
        "outputs": [],
        "source": source(text),
    }


TEXT = {
    "en": {
        "title": "From Teacher to Learner: A Complete Imitation-Learning Pipeline",
        "edition_student": "Student edition",
        "edition_solution": "Solutions edition",
        "intro": """
            **Audience:** strong high-school students who know Python, arrays, and basic algebra.

            In this lab you will build and inspect the four missing boxes after the simulator:
            a scripted expert, a demonstration database, behavior cloning, and closed-loop evaluation.
            This is not a toy diagram—the same data contracts and experimental safeguards appear in
            real robotics research.

            **By the end, you can:**

            1. explain a finite-state expert and inspect where it fails;
            2. store `(observation, action)` demonstrations without mixing episodes;
            3. train an MLP by minimizing mean-squared action error;
            4. distinguish validation loss from closed-loop success rate;
            5. report a success rate with uncertainty and reproducible seeds.

            Look for **🧠 Checkpoint** questions and **🛠 Your turn** cells. A normal first pass takes
            90–120 minutes; the 45-minute mentor meeting uses selected sections.
        """,
        "map_title": "## 0. The map: five boxes, four contracts",
        "map_text": """
            The simulator emits a 22-number state. The expert labels that state with a 4-number
            action. The database preserves pairs and episode boundaries. The learner predicts an
            action. Evaluation returns the policy to the simulator—because low homework error does
            not guarantee that a robot can recover from its own mistakes.
        """,
        "setup_title": "## 1. Setup and reproducibility",
        "setup_text": """
            Run from the project root with the Python 3.11 environment described in `README.md`.
            `RUN_SIM = False` lets you read and complete pure-Python exercises without PyBullet;
            change it to `True` when the environment is installed. We reserve seeds 0–9999 for data
            and seeds 10000+ for evaluation.
        """,
        "contract_title": "## 2. Contract A — observation in, action out",
        "contract_text": """
            The flat state is `main observation (19) + desired goal (3) = 22`. The object position
            is already in slots `7:10`, so repeating `achieved_goal` would add duplicate information.
            The action is `[dx, dy, dz, gripper]`, clipped to `[-1, 1]`.

            **🛠 Your turn 1.** Implement `decode_positions`. Do not use magic copies of the goal;
            read it from the final three entries.
        """,
        "contract_hint": "Hint: end effector is `0:3`, object is `7:10`, goal is `19:22`.",
        "expert_title": "## 3. The scripted expert: a finite-state machine",
        "expert_text": """
            One equation drives every movement phase:

            $$a_{xyz}=\\operatorname{clip}(k(p_{target}-p_{ee}),-1,1).$$

            The hard part is not this equation—it is deciding *which target is active*. The expert
            uses seven phases: approach, descend, grasp, lift, transport, place, and hold. It is
            allowed to see exact object position because it is a data-labeling teacher, not the final
            learned policy.

            **🛠 Your turn 2.** Implement one proportional movement step.
        """,
        "expert_hint": "Remember to return four `float32` values and clip all four action slots.",
        "expert_viz_title": "### 3.1 Diagnose behavior, do not just watch it",
        "expert_viz_text": """
            A useful expert plot answers three questions at once: *where did the hand move, which
            phase controlled it, and did the object approach the goal?* Set `RUN_SIM=True` to collect
            an episode. The second panel is often more diagnostic than a 3-D animation.
        """,
        "expert_question": "**🧠 Checkpoint:** If the LIFT phase lasts forever, which target might accidentally be moving upward together with the object?",
        "db_title": "## 4. Contract B — episodes into SQLite",
        "db_text": """
            A timestep is one supervised example; an episode is one complete attempt. We store both
            levels. The write is atomic: either every step of an episode is saved, or none is. Failed
            attempts remain in the database for expert debugging, while the training query selects
            successful demonstrations only. Rare one-step episodes that start already at the goal
            count as task successes but are marked unusable because they teach no grasp behavior.

            **🛠 Your turn 3.** Before running the query, predict the two shapes returned for 50
            episodes of roughly 70 steps each.
        """,
        "sql_question": "**🧠 Checkpoint:** Why is `WHERE success = 1` appropriate for basic BC, and what information do we lose by filtering failures?",
        "dashboard_title": "### 4.1 Dataset dashboard",
        "dashboard_text": """
            Never train before looking at the data. The dashboard checks phase imbalance, action
            saturation, state coverage, and episode length. A phase with very few examples can be
            ignored by ordinary MSE; the training helper can compensate with bounded inverse-frequency
            weights.
        """,
        "learner_title": "## 5. Contract C — behavior cloning",
        "learner_text": """
            The MLP computes `22 → hidden → hidden → 4`. Training minimizes

            $$\\mathcal{L}(\\theta)=\\frac1N\\sum_i\\|\\pi_\\theta(o_i)-a_i^*\\|_2^2.$$

            Inputs are standardized using **training data only**, and the mean/std travel inside the
            checkpoint. The final `tanh` guarantees actions remain in `[-1,1]`.

            **🛠 Your turn 4.** Count the trainable parameters without using a built-in summary tool.
        """,
        "split_title": "### 5.1 The subtle rule: split by episode",
        "split_text": """
            Adjacent timesteps are almost copies. A random timestep split would put near-identical
            states from one trajectory on both sides and make validation look too easy. We therefore
            choose whole episode IDs for train or validation—never both.
        """,
        "train_title": "### 5.2 Train and read the curves",
        "train_text": """
            Start small: 10–50 demos and 30 epochs are enough for a smoke test. A falling loss proves
            the optimization works. It does **not** prove the robot succeeds in closed loop.
        """,
        "prediction_title": "### 5.3 What errors does the network make?",
        "prediction_text": """
            Predicted-versus-expert scatter plots expose bias that one average loss can hide. Perfect
            predictions lie on the diagonal. Pay special attention to the gripper action: a smooth
            regressor may average `open (+1)` and `close (-1)` into an unhelpful value near zero.
        """,
        "eval_title": "## 6. Contract D — closed-loop evaluation",
        "eval_text": """
            In open-loop validation, the learner receives states visited by the expert. In closed-loop
            evaluation, it receives states caused by its *own earlier actions*. Tiny errors can move it
            off the demonstration path and compound. We evaluate on fresh seeds and report a 95%
            Wilson interval, not only a percentage.
        """,
        "eval_question": "**🧠 Checkpoint:** Which would you trust more: 8/10 successes or 72/100? Both are estimates—compare their interval widths.",
        "experiment_title": "## 7. Turn the pipeline into a research experiment",
        "experiment_text": """
            The main experiment varies successful demonstration count `[10, 25, 50, 100, 200]`.
            For each count, train three random seeds, evaluate on the same held-out seed set, and plot
            mean ± standard deviation. A second experiment varies hidden width while holding the data
            fixed. Change one independent variable at a time.

            **🛠 Your turn 5.** Complete `summarize_trials` so it returns mean and sample standard
            deviation for each setting.
        """,
        "failure_title": "## 8. Failure analysis and the DAgger extension",
        "failure_text": """
            Behavior cloning has a distribution-shift problem: the expert labels states on good paths,
            but the learner may visit different states. DAgger repeats three operations: let the learner
            visit states, ask the expert for the correct actions there, and aggregate those new labels
            into the dataset.

            Before trying DAgger, select one failed evaluation seed and record: first phase-like mistake,
            distance to object/goal over time, action disagreement, and a falsifiable explanation. A
            useful failure report ends with a proposed test, not “the network is bad.”
        """,
        "finish_title": "## 9. Definition of done",
        "finish_text": """
            Your first complete result should include:

            - expert successes / attempts and phase-duration plots;
            - successful episode count and state-action sample count;
            - train/validation loss with an episode-disjoint split;
            - learner successes / 100 fresh episodes with a 95% interval;
            - three training seeds for every final comparison;
            - one analyzed failure case and the next experiment it motivates.

            **Exit ticket:** explain in one sentence why “my validation loss is low” is not yet a
            robotics result.
        """,
    },
    "zh": {
        "title": "从老师到学习器：完整的机器人模仿学习 Pipeline",
        "edition_student": "学生练习版",
        "edition_solution": "答案版",
        "intro": """
            **适合对象：** 会 Python、数组和基础代数，且愿意深入思考的高中生。

            这份实验会补齐 simulator 后面的四个部件：脚本专家、示范数据库、行为克隆和闭环评估。
            这不只是教学示意图；其中的数据契约、随机种子、验证集和不确定性报告，都是机器人研究中
            真正使用的习惯。

            **完成后你应该能够：**

            1. 解释有限状态机 expert，并定位它在哪个阶段失败；
            2. 保存 `(observation, action)`，同时保留 episode 边界；
            3. 用均方误差训练 MLP 模仿 expert；
            4. 区分 validation loss 与闭环 success rate；
            5. 用新 seed 和置信区间规范地报告成功率。

            请留意 **🧠 检查点** 与 **🛠 轮到你**。完整做一遍约需 90–120 分钟；45 分钟导师会议
            只选择其中关键部分。
        """,
        "map_title": "## 0. 全局地图：五个方框、四个接口",
        "map_text": """
            simulator 输出 22 维状态，expert 给这个状态标注 4 维动作，database 保存配对数据与
            episode 边界，learner 预测动作，evaluation 再把 learner 放回 simulator。最后一步不能省略：
            “作业题误差小”不代表机器人能从自己的错误中恢复。
        """,
        "setup_title": "## 1. 环境与可复现性",
        "setup_text": """
            请按照 `README.md` 在项目根目录使用 Python 3.11 环境。`RUN_SIM = False` 时，即使尚未装好
            PyBullet，也能阅读并完成纯 Python 练习；环境可用后改为 `True`。我们约定 0–9999 用于
            收集数据，10000 以上用于评估，防止测试场景泄漏进训练。
        """,
        "contract_title": "## 2. 接口 A：输入 observation，输出 action",
        "contract_text": """
            扁平状态为 `主 observation (19) + desired goal (3) = 22`。object 已在 `7:10`，因此不再
            重复拼接 `achieved_goal`。动作为 `[dx, dy, dz, gripper]`，每一维都限制在 `[-1, 1]`。

            **🛠 轮到你 1：** 实现 `decode_positions`。不要复制 goal；从最后三维读取它。
        """,
        "contract_hint": "提示：末端执行器为 `0:3`，物体为 `7:10`，目标为 `19:22`。",
        "expert_title": "## 3. 脚本专家：有限状态机",
        "expert_text": """
            每个移动阶段都由同一个公式驱动：

            $$a_{xyz}=\\operatorname{clip}(k(p_{target}-p_{ee}),-1,1).$$

            真正困难的不是公式，而是决定“当前 target 是谁”。expert 有七个阶段：approach、descend、
            grasp、lift、transport、place、hold。它可以读取 object 真值位置，因为它只是生成标签的老师，
            不是最终部署的 learned policy。

            **🛠 轮到你 2：** 实现一步 proportional movement。
        """,
        "expert_hint": "记得返回 4 个 `float32` 数，并把四个动作都 clip 到允许范围。",
        "expert_viz_title": "### 3.1 诊断行为，而不是只看动画",
        "expert_viz_text": """
            一张有用的 expert 图应同时回答：手走到哪里、哪个 phase 在控制、object 是否接近 goal。
            将 `RUN_SIM=True` 后收集一条 episode。很多时候第二张距离曲线比 3D 动画更容易发现 bug。
        """,
        "expert_question": "**🧠 检查点：** 如果 LIFT 永远结束不了，是否可能 target 随 object 一起不断升高？",
        "db_title": "## 4. 接口 B：把 episode 写入 SQLite",
        "db_text": """
            一个 timestep 是一个监督学习样本；一个 episode 是一次完整尝试。数据库同时保存两层。
            每条 episode 的写入是原子的：要么所有 step 都成功保存，要么一条也不留。失败尝试仍可用于
            调试 expert，但基本 BC 的训练查询只选择成功示范。极少数“初始位置已经满足目标”的一步成功
            仍计入 task success，但会被标成不可训练，因为其中没有任何抓取行为。

            **🛠 轮到你 3：** 执行 query 前，先预测 50 个、每个约 70 步的 episode 会返回什么 shape。
        """,
        "sql_question": "**🧠 检查点：** 为什么基础 BC 适合 `WHERE success = 1`？过滤失败数据又丢掉了什么信息？",
        "dashboard_title": "### 4.1 数据集体检图",
        "dashboard_text": """
            训练前必须看数据。下面的 dashboard 检查 phase 是否失衡、action 是否大量饱和、state 覆盖范围
            和 episode 长度。普通 MSE 可能忽略样本很少的阶段；训练函数可使用有上下限的逆频率权重补偿。
        """,
        "learner_title": "## 5. 接口 C：行为克隆",
        "learner_text": """
            MLP 的尺寸为 `22 → hidden → hidden → 4`，目标函数是

            $$\\mathcal{L}(\\theta)=\\frac1N\\sum_i\\|\\pi_\\theta(o_i)-a_i^*\\|_2^2.$$

            observation 只用训练集统计量标准化，mean/std 会和模型一起存入 checkpoint；最后的 `tanh`
            保证 action 始终在 `[-1,1]`。

            **🛠 轮到你 4：** 不调用 summary 工具，自己计算 trainable parameter 数量。
        """,
        "split_title": "### 5.1 容易忽略的规则：按 episode 划分",
        "split_text": """
            相邻 timestep 几乎一样。若随机打散所有 step，同一条轨迹的近似副本会同时出现在 train 和
            validation，使验证结果虚高。因此每个 episode ID 必须完整属于一边，不能跨两边。
        """,
        "train_title": "### 5.2 训练并读懂曲线",
        "train_text": """
            先做小规模 smoke test：10–50 条 demo、30 epochs 足以检查优化是否工作。loss 下降只证明
            “更会模仿”，并不证明机器人闭环执行时成功。
        """,
        "prediction_title": "### 5.3 网络究竟错在哪里？",
        "prediction_text": """
            prediction-vs-expert 散点图能暴露平均 loss 隐藏的偏差；完美预测应落在对角线上。特别关注
            gripper：平滑回归器可能把 open (+1) 和 close (-1) 平均成几乎不动的 0。
        """,
        "eval_title": "## 6. 接口 D：闭环评估",
        "eval_text": """
            open-loop validation 给 learner 的是 expert 走过的状态；closed-loop evaluation 给它的是由
            自己之前动作造成的状态。小误差会把机器人带离示范轨迹并逐步累积。因此必须用全新 seed，
            并报告 95% Wilson interval，而不只是一个百分比。
        """,
        "eval_question": "**🧠 检查点：** 8/10 与 72/100 哪个更可信？两者都是估计值，请比较置信区间宽度。",
        "experiment_title": "## 7. 把 pipeline 变成研究实验",
        "experiment_text": """
            主实验改变成功 demo 数 `[10, 25, 50, 100, 200]`。每个数量训练三个随机 seed，在同一组
            held-out seeds 上评估，画 mean ± standard deviation。第二个实验固定数据，只改变 hidden width。
            每次只改变一个 independent variable。

            **🛠 轮到你 5：** 完成 `summarize_trials`，为每个 setting 返回 mean 与 sample standard deviation。
        """,
        "failure_title": "## 8. 失败分析与 DAgger 扩展",
        "failure_text": """
            行为克隆有 distribution shift：expert 只标注良好路径上的状态，而 learner 可能走到别处。
            DAgger 重复三件事：让 learner 访问状态、请 expert 为这些状态给正确动作、把新标签加入数据集。

            尝试 DAgger 前，先选择一个失败 seed，记录首次明显错误、object/goal 距离、动作分歧和一个可证伪
            的解释。好的 failure report 以“下一步如何验证”结束，而不是“网络不好”。
        """,
        "finish_title": "## 9. 完成标准",
        "finish_text": """
            第一份完整结果应包含：

            - expert 成功数 / 总尝试数与 phase duration 图；
            - 成功 episode 数与 state-action 样本数；
            - episode 不重叠的 train/validation loss；
            - learner 在 100 个全新 episode 上的成功数与 95% 区间；
            - 每个最终对比使用三个训练 seed；
            - 一个已分析的失败案例，以及它引出的下一项实验。

            **离场题：** 用一句话解释为什么“validation loss 很低”还不能算机器人实验结果。
        """,
    },
}


def exercise_decode(solution: bool, lang: str) -> str:
    if solution:
        return """
            def decode_positions(flat_obs):
                flat_obs = np.asarray(flat_obs, dtype=np.float32)
                if flat_obs.shape != (OBS_DIM,):
                    raise ValueError(f"expected ({OBS_DIM},), got {flat_obs.shape}")
                return flat_obs[0:3], flat_obs[7:10], flat_obs[19:22]

            toy = np.arange(OBS_DIM, dtype=np.float32)
            ee, obj, goal = decode_positions(toy)
            assert np.array_equal(ee, [0, 1, 2])
            assert np.array_equal(obj, [7, 8, 9])
            assert np.array_equal(goal, [19, 20, 21])
            print("✓ decode_positions passed")
        """
    todo = "TODO：请填写三个切片" if lang == "zh" else "TODO: fill in the three slices"
    return f"""
        def decode_positions(flat_obs):
            flat_obs = np.asarray(flat_obs, dtype=np.float32)
            # {todo}
            ee = None
            obj = None
            goal = None
            return ee, obj, goal

        toy = np.arange(OBS_DIM, dtype=np.float32)
        answer = decode_positions(toy)
        print("current answer:", answer)
    """


def exercise_move(solution: bool, lang: str) -> str:
    if solution:
        return """
            def proportional_action(ee, target, gripper, gain=20.0):
                ee = np.asarray(ee, dtype=np.float32)
                target = np.asarray(target, dtype=np.float32)
                move = np.clip(gain * (target - ee), -1.0, 1.0)
                return np.clip(np.r_[move, gripper], -1.0, 1.0).astype(np.float32)

            test_action = proportional_action([0, 0, 0.2], [0.1, -0.1, 0.25], -1)
            assert test_action.shape == (4,) and test_action.dtype == np.float32
            assert np.all(np.abs(test_action) <= 1)
            print("action:", test_action, "✓")
        """
    todo = "TODO：计算 target - ee、乘 gain、clip、拼接 gripper" if lang == "zh" else "TODO: target - ee, multiply by gain, clip, append gripper"
    return f"""
        def proportional_action(ee, target, gripper, gain=20.0):
            # {todo}
            return None

        print("current answer:", proportional_action([0, 0, 0.2], [0.1, -0.1, 0.25], -1))
    """


def exercise_params(solution: bool, lang: str) -> str:
    if solution:
        return """
            def count_parameters(model):
                return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

            demo_model = GraspPolicy(hidden_sizes=(128, 128))
            expected = (22 * 128 + 128) + (128 * 128 + 128) + (128 * 4 + 4)
            assert count_parameters(demo_model) == expected
            print(f"trainable parameters: {count_parameters(demo_model):,} ✓")
        """
    todo = "TODO：遍历 model.parameters() 并累加 numel()" if lang == "zh" else "TODO: iterate over model.parameters() and add numel()"
    return f"""
        def count_parameters(model):
            # {todo}
            return None

        demo_model = GraspPolicy(hidden_sizes=(128, 128)) if TORCH_AVAILABLE else None
        print("current answer:", count_parameters(demo_model) if demo_model else "install torch first")
    """


def exercise_summary(solution: bool, lang: str) -> str:
    if solution:
        return """
            def summarize_trials(rows, x_key):
                summary = {}
                for x in sorted({row[x_key] for row in rows}):
                    values = np.array([row["success_rate"] for row in rows if row[x_key] == x])
                    summary[x] = {"mean": values.mean(), "std": values.std(ddof=1)}
                return summary

            toy_trials = [
                {"demos": 10, "success_rate": 0.20}, {"demos": 10, "success_rate": 0.30},
                {"demos": 10, "success_rate": 0.25}, {"demos": 50, "success_rate": 0.65},
                {"demos": 50, "success_rate": 0.70}, {"demos": 50, "success_rate": 0.75},
            ]
            trial_summary = summarize_trials(toy_trials, "demos")
            print(trial_summary)
            assert np.isclose(trial_summary[10]["mean"], 0.25)
        """
    todo = "TODO：按 x_key 分组，并用 ddof=1 计算样本标准差" if lang == "zh" else "TODO: group by x_key and use ddof=1 for sample standard deviation"
    return f"""
        def summarize_trials(rows, x_key):
            # {todo}
            return {{}}

        toy_trials = [
            {{"demos": 10, "success_rate": 0.20}}, {{"demos": 10, "success_rate": 0.30}},
            {{"demos": 10, "success_rate": 0.25}}, {{"demos": 50, "success_rate": 0.65}},
            {{"demos": 50, "success_rate": 0.70}}, {{"demos": 50, "success_rate": 0.75}},
        ]
        print("current answer:", summarize_trials(toy_trials, "demos"))
    """


def build_notebook(lang: str, solution: bool) -> dict:
    t = TEXT[lang]
    edition = t["edition_solution"] if solution else t["edition_student"]
    label = "solutions" if solution else "student"
    cells: list[dict] = []
    add_m = lambda key, body: cells.append(markdown(body, f"{key}-{lang}-{label}"))
    add_c = lambda key, body, tags=None: cells.append(code(body, f"{key}-{lang}-{label}", tags))

    add_m("title", f"# {t['title']}\n\n> **{edition} · Notebook 2**\n\n{t['intro']}")
    add_m("map", f"{t['map_title']}\n\n{t['map_text']}")
    add_c("pipeline-plot", """
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

        names = ["SIMULATOR", "EXPERT", "DATABASE", "LEARNER", "EVALUATION"]
        subtitles = ["22-D state", "4-D label", "episodes", "MLP policy", "fresh seeds"]
        colors = ["#0f766e", "#2563eb", "#7c3aed", "#db2777", "#ea580c"]
        fig, ax = plt.subplots(figsize=(13, 2.6))
        ax.set_xlim(-0.4, 12.6); ax.set_ylim(-0.2, 2.4); ax.axis("off")
        for i, (name, sub, color) in enumerate(zip(names, subtitles, colors)):
            x = 2.55 * i
            box = FancyBboxPatch((x, 0.55), 2.0, 1.2, boxstyle="round,pad=0.08,rounding_size=0.12",
                                 facecolor=color, edgecolor="none", alpha=0.95)
            ax.add_patch(box)
            ax.text(x + 1, 1.25, name, ha="center", va="center", color="white", weight="bold", fontsize=11)
            ax.text(x + 1, 0.88, sub, ha="center", va="center", color="white", fontsize=9)
            if i < len(names) - 1:
                ax.add_patch(FancyArrowPatch((x + 2.05, 1.15), (x + 2.48, 1.15), arrowstyle="-|>",
                                             mutation_scale=14, color="#475569", lw=1.8))
        ax.add_patch(FancyArrowPatch((11.2, 0.48), (1.0, 0.48), arrowstyle="-|>",
                                     connectionstyle="arc3,rad=-0.16", mutation_scale=14,
                                     color="#475569", lw=1.4))
        ax.text(6.1, -0.02, "closed-loop test in the simulator", ha="center", color="#475569", fontsize=9)
        plt.show()
    """)

    add_m("setup", f"{t['setup_title']}\n\n{t['setup_text']}")
    add_c("imports", """
        from pathlib import Path
        import sys
        import sqlite3
        import numpy as np
        import matplotlib.pyplot as plt

        PROJECT_ROOT = Path.cwd()
        if not (PROJECT_ROOT / "pyproject.toml").exists():
            PROJECT_ROOT = PROJECT_ROOT.parent
        sys.path.insert(0, str(PROJECT_ROOT / "src"))

        from ilgrasp.sim import OBS_DIM, ACTION_DIM
        from ilgrasp.experts import ScriptedExpert, collect_expert_episode
        from ilgrasp.data import DemoDatabase
        from ilgrasp.evaluation import wilson_interval

        try:
            import torch
            from ilgrasp.policies import GraspPolicy, NumpyPolicy, train_behavior_cloning
            TORCH_AVAILABLE = True
        except ImportError:
            TORCH_AVAILABLE = False

        RUN_SIM = False  # Change to True after installing panda-gym.
        DB_PATH = PROJECT_ROOT / "artifacts" / "notebook_demos.db"
        MODEL_PATH = PROJECT_ROOT / "artifacts" / "notebook_policy.pt"
        (PROJECT_ROOT / "artifacts").mkdir(exist_ok=True)

        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update({"figure.dpi": 120, "axes.titleweight": "bold", "axes.spines.top": False,
                             "axes.spines.right": False, "grid.alpha": 0.22})
        print(f"project: {PROJECT_ROOT}")
        print(f"torch available: {TORCH_AVAILABLE} | run simulator: {RUN_SIM}")
    """)

    add_m("contract", f"{t['contract_title']}\n\n{t['contract_text']}\n\n> {t['contract_hint']}")
    add_c("exercise-decode", exercise_decode(solution, lang), ["solution"] if solution else ["exercise"])

    add_m("expert", f"{t['expert_title']}\n\n{t['expert_text']}\n\n> {t['expert_hint']}")
    add_c("exercise-move", exercise_move(solution, lang), ["solution"] if solution else ["exercise"])
    add_c("state-machine", """
        phase_order = ["approach", "descend", "grasp", "lift", "transport", "place", "hold"]
        phase_targets = ["above object", "at object", "at object", "safe z", "above goal", "at goal", "at goal"]
        phase_grip = ["open", "open", "close", "close", "close", "close", "close"]
        print(f"{'phase':<12} {'target':<14} gripper")
        print("-" * 38)
        for phase, target, grip in zip(phase_order, phase_targets, phase_grip):
            print(f"{phase:<12} {target:<14} {grip}")
    """)

    add_m("expert-viz", f"{t['expert_viz_title']}\n\n{t['expert_viz_text']}")
    add_c("collect-one", """
        expert_episode = None
        if RUN_SIM:
            from ilgrasp.sim import GraspEnv
            with GraspEnv() as env:
                expert_episode = collect_expert_episode(env, ScriptedExpert(), seed=7)
            print("success:", expert_episode.success, "| steps:", len(expert_episode))
        else:
            print("Set RUN_SIM=True to collect and visualize a real expert episode.")
    """)
    add_c("expert-plot", """
        if expert_episode is not None:
            obs = expert_episode.observations
            ee, obj, goal = obs[:, 0:3], obs[:, 7:10], obs[:, 19:22]
            palette = dict(zip(phase_order, plt.cm.turbo(np.linspace(0.05, 0.95, len(phase_order)))))
            colors = [palette[p] for p in expert_episode.phases]

            fig = plt.figure(figsize=(12, 4.8))
            ax = fig.add_subplot(121, projection="3d")
            for phase in phase_order:
                mask = np.array(expert_episode.phases) == phase
                if mask.any():
                    ax.scatter(*ee[mask].T, s=18, color=palette[phase], label=phase, alpha=0.85)
            ax.plot(*ee.T, color="#334155", lw=1, alpha=.45)
            ax.scatter(*obj[0], marker="s", s=90, color="#111827", label="object start")
            ax.scatter(*goal[0], marker="*", s=160, color="#fbbf24", edgecolor="#111827", label="goal")
            ax.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", title="Expert end-effector path")
            ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0, .94))

            ax2 = fig.add_subplot(122)
            steps = np.arange(len(obs))
            ax2.plot(steps, np.linalg.norm(ee - obj, axis=1), label="gripper → object", lw=2)
            ax2.plot(steps, np.linalg.norm(obj - goal, axis=1), label="object → goal", lw=2)
            for phase in phase_order:
                mask = np.array(expert_episode.phases) == phase
                if mask.any():
                    ax2.axvspan(np.flatnonzero(mask)[0], np.flatnonzero(mask)[-1] + 1,
                                color=palette[phase], alpha=.10)
            ax2.set(xlabel="step", ylabel="distance (m)", title="Distances reveal stuck phases")
            ax2.legend()
            fig.tight_layout(); plt.show()
        else:
            print("No episode yet.")
    """)
    add_m("expert-check", t["expert_question"])

    add_m("database", f"{t['db_title']}\n\n{t['db_text']}")
    add_c("database-work", """
        if expert_episode is not None:
            with DemoDatabase(DB_PATH) as db:
                episode_id = db.save_episode(expert_episode)
                print("saved episode id:", episode_id)
                print("database summary:", db.summary())
                if expert_episode.success:
                    training_data = db.load_training_data(successful_only=True)
                    print("observations:", training_data.observations.shape)
                    print("actions     :", training_data.actions.shape)
        else:
            print("After collecting an episode, this cell performs the SQLite round trip.")
    """)
    add_c("sql", """
        if DB_PATH.exists():
            with sqlite3.connect(DB_PATH) as connection:
                rows = connection.execute('''
                    SELECT success, COUNT(*) AS episodes, ROUND(AVG(n_steps), 1) AS mean_steps
                    FROM episodes GROUP BY success ORDER BY success DESC
                ''').fetchall()
            print("success | episodes | mean steps")
            for row in rows: print(f"{row[0]:7d} | {row[1]:8d} | {row[2]:10.1f}")
        else:
            print("Database not created yet.")
    """)
    add_m("sql-check", t["sql_question"])

    add_m("dashboard", f"{t['dashboard_title']}\n\n{t['dashboard_text']}")
    add_c("dashboard-plot", """
        if 'training_data' in globals():
            data = training_data
            fig, axes = plt.subplots(2, 2, figsize=(11, 7))
            labels, counts = np.unique(data.phases, return_counts=True)
            axes[0, 0].bar(labels, counts, color=plt.cm.turbo(np.linspace(.1, .9, len(labels))))
            axes[0, 0].tick_params(axis="x", rotation=35); axes[0, 0].set_title("Samples per expert phase")
            axes[0, 0].set_ylabel("timesteps")

            action_names = ["dx", "dy", "dz", "gripper"]
            axes[0, 1].boxplot([data.actions[:, i] for i in range(4)], tick_labels=action_names,
                               patch_artist=True, boxprops=dict(facecolor="#bfdbfe"))
            axes[0, 1].set_title("Action distribution"); axes[0, 1].set_ylim(-1.1, 1.1)

            sc = axes[1, 0].scatter(data.observations[:, 7], data.observations[:, 8],
                                    c=data.observations[:, 9], cmap="viridis", s=9, alpha=.55)
            axes[1, 0].set(xlabel="object x", ylabel="object y", title="Object-state coverage")
            fig.colorbar(sc, ax=axes[1, 0], label="object z")

            ids, lengths = np.unique(data.episode_ids, return_counts=True)
            axes[1, 1].hist(lengths, bins=min(12, len(lengths)), color="#7c3aed", alpha=.78)
            axes[1, 1].axvline(lengths.mean(), color="#111827", ls="--", label=f"mean={lengths.mean():.1f}")
            axes[1, 1].set(xlabel="steps", ylabel="episodes", title="Successful episode lengths")
            axes[1, 1].legend()
            fig.suptitle("Demonstration dataset health check", fontsize=15, weight="bold")
            fig.tight_layout(); plt.show()
        else:
            print("Load successful training_data first.")
    """)

    add_m("learner", f"{t['learner_title']}\n\n{t['learner_text']}")
    add_c("exercise-params", exercise_params(solution, lang), ["solution"] if solution else ["exercise"])
    add_m("split", f"{t['split_title']}\n\n{t['split_text']}")
    add_c("split-demo", """
        if 'training_data' in globals():
            ids = np.unique(training_data.episode_ids)
            rng = np.random.default_rng(0)
            shuffled = rng.permutation(ids)
            n_val = max(1, round(.2 * len(ids))) if len(ids) >= 2 else 0
            validation_ids, train_ids = shuffled[:n_val], shuffled[n_val:]
            print("train episode ids     :", np.sort(train_ids))
            print("validation episode ids:", np.sort(validation_ids))
            print("overlap:", set(train_ids) & set(validation_ids))
        else:
            print("Load training_data first.")
    """)

    add_m("train", f"{t['train_title']}\n\n{t['train_text']}")
    add_c("train-code", """
        trained = None
        if TORCH_AVAILABLE and 'training_data' in globals() and len(np.unique(training_data.episode_ids)) >= 2:
            trained = train_behavior_cloning(
                training_data, hidden_sizes=(128, 128), epochs=30, batch_size=256,
                learning_rate=1e-3, validation_fraction=.2, seed=0,
            )
            trained.model.save(MODEL_PATH)
            print("saved:", MODEL_PATH)
        else:
            print("Need torch and at least two successful episodes before training.")
    """)
    add_c("loss-plot", """
        if trained is not None:
            fig, ax = plt.subplots(figsize=(7.5, 4.3))
            epochs = np.arange(1, len(trained.train_loss) + 1)
            ax.plot(epochs, trained.train_loss, lw=2.3, label="train", color="#2563eb")
            val = np.asarray(trained.validation_loss)
            if np.isfinite(val).any(): ax.plot(epochs, val, lw=2.3, label="validation", color="#ea580c")
            ax.set(xlabel="epoch", ylabel="weighted action MSE", title="Behavior cloning learning curves")
            ax.set_yscale("log"); ax.legend(); plt.show()
        else:
            print("Train a model first.")
    """)

    add_m("prediction", f"{t['prediction_title']}\n\n{t['prediction_text']}")
    add_c("prediction-plot", """
        if trained is not None:
            mask = np.isin(training_data.episode_ids, trained.validation_episode_ids)
            x = torch.as_tensor(training_data.observations[mask], dtype=torch.float32)
            with torch.inference_mode(): predicted = trained.model(x).numpy()
            truth = training_data.actions[mask]
            names = ["dx", "dy", "dz", "gripper"]
            fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharex=True, sharey=True)
            for i, ax in enumerate(axes):
                ax.scatter(truth[:, i], predicted[:, i], s=9, alpha=.35, color="#2563eb")
                ax.plot([-1, 1], [-1, 1], "--", color="#ef4444", lw=1)
                ax.set_title(names[i]); ax.set_xlabel("expert")
            axes[0].set_ylabel("prediction")
            fig.suptitle("Held-out action predictions", weight="bold"); fig.tight_layout(); plt.show()
        else:
            print("Train a model first.")
    """)

    add_m("evaluation", f"{t['eval_title']}\n\n{t['eval_text']}")
    add_c("interval-demo", """
        for successes, trials in [(8, 10), (72, 100)]:
            low, high = wilson_interval(successes, trials)
            print(f"{successes:3d}/{trials:<3d} = {successes/trials:5.1%} | 95% CI [{low:5.1%}, {high:5.1%}] | width {high-low:5.1%}")
    """)
    add_c("evaluate-code", """
        evaluation = None
        if RUN_SIM and trained is not None:
            from ilgrasp.sim import GraspEnv
            from ilgrasp.evaluation import evaluate_policy
            with GraspEnv() as env:
                evaluation = evaluate_policy(env, NumpyPolicy(trained.model), n_episodes=20, seed_start=10_000)
            print(evaluation.as_dict())
        else:
            print("Set RUN_SIM=True and train a model. Use 20 episodes for a smoke test, then 100 for reporting.")
    """)
    add_c("evaluation-plot", """
        if evaluation is not None:
            rate = evaluation.success_rate
            low, high = evaluation.confidence_interval
            fig, ax = plt.subplots(figsize=(5.8, 4.2))
            ax.bar(["BC policy"], [rate], color="#0f766e", width=.55)
            ax.errorbar([0], [rate], yerr=[[rate-low], [high-rate]], fmt="none", color="#111827", capsize=8, lw=2)
            ax.text(0, rate/2, f"{int(evaluation.successes.sum())}/{evaluation.n_episodes}", ha="center", color="white", weight="bold", fontsize=15)
            ax.set(ylabel="success rate", ylim=(0, 1.05), title="Closed-loop performance (95% CI)")
            ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
            plt.show()
        else:
            print("Evaluate a model first.")
    """)
    add_m("evaluation-check", t["eval_question"])

    add_m("experiment", f"{t['experiment_title']}\n\n{t['experiment_text']}")
    add_c("exercise-summary", exercise_summary(solution, lang), ["solution"] if solution else ["exercise"])
    add_c("experiment-plot", """
        # Replace these illustrative values with scripts/run_experiments.py results.
        illustrative = {10: (0.18, .06), 25: (.37, .07), 50: (.55, .05), 100: (.68, .04), 200: (.74, .03)}
        xs = np.array(list(illustrative)); means = np.array([illustrative[x][0] for x in xs]); stds = np.array([illustrative[x][1] for x in xs])
        fig, ax = plt.subplots(figsize=(7.2, 4.3))
        ax.plot(xs, means, marker="o", lw=2.4, color="#7c3aed")
        ax.fill_between(xs, means-stds, means+stds, color="#7c3aed", alpha=.17, label="± 1 std (3 seeds)")
        ax.set(xlabel="successful demonstrations", ylabel="success rate", ylim=(0, 1), title="Data-efficiency curve — illustrative only")
        ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0)); ax.legend(); plt.show()
        print("Do not report these illustrative numbers as experimental results.")
    """)

    add_m("failure", f"{t['failure_title']}\n\n{t['failure_text']}")
    add_c("failure-template", """
        failure_report = {
            "seed": None,
            "first_visible_error_step": None,
            "observation": "",
            "hypothesis": "",
            "next_test": "",
        }
        failure_report
    """)
    add_m("finish", f"{t['finish_title']}\n\n{t['finish_text']}")

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3 (ilgrasp)", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "ilgrasp": {"language": lang, "edition": label, "paired_notebook": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    targets = [
        ("en", False, "02_full_pipeline_student_en.ipynb"),
        ("zh", False, "02_full_pipeline_student_zh.ipynb"),
        ("en", True, "02_full_pipeline_solutions_en.ipynb"),
        ("zh", True, "02_full_pipeline_solutions_zh.ipynb"),
    ]
    for lang, solution, filename in targets:
        path = NOTEBOOK_DIR / filename
        path.write_text(json.dumps(build_notebook(lang, solution), ensure_ascii=False, indent=1) + "\n")
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
