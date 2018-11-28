import np from 'numjs';
import * as _ from 'lodash';
import {Random} from './utils';

function softmax(x) {
    let probs = np.exp(x.subtract(np.max(x)));
    probs = probs.divide(np.sum(probs));
    return probs;
}

class TreeNode {
    // 定义蒙特卡洛树的节点
    // 每一个节点保存自己的Q值，先验概率P和访问计数N
    constructor(parent, prior_p) {
        this._parent = parent;
        this._children = {};
        this._n_visits = 0;
        this._Q = 0;
        this._u = 0;
        this._P = prior_p;
    }

    expand(action_priors) {
        // 扩展，蒙特卡洛树的基本操作之一。
        // 利用先验概率扩展，一次扩展所有的子节点，标准MCTS一次只扩展一个节点
        // 先验概率是由神经网络得到的。
        for (let [action, prob] of action_priors) {
            if (!this._children[action]) {
                this._children[action] = new TreeNode(this, prob);
            }
        }
    }

    select(c_puct) {
        // 选择，蒙特卡洛树的基本操作之一，根据论文的描述，选择当前最大的Q+u值的那个动作
        // 返回的是一个元组，包括选择的动作和孩子节点

        let keys = Object.keys(this._children);

        let move = _.maxBy(keys, (move) => {
            return this._children[move].get_value(c_puct);
        });
        let action = parseInt(move);
        let node = this._children[move];
        return {action, node};

    }

    update(leaf_value) {
        // 更新，其实也就是backup，同样是蒙特卡洛树的基本操作之一，
        // 从叶节点更新自己的评价
        // leaf_value是当前玩家视角下的叶节点价值

        // 更新包括计数加一，Q值更新
        this._n_visits += 1;
        // Q值采用滑动平均方法更新
        this._Q += 1.0 * (leaf_value - this._Q) / this._n_visits;
    }

    update_recursive(leaf_value) {
        // 递归更新所有祖先的相关信息，也就是递归调用update
        if (this._parent) {
            this._parent.update_recursive(-leaf_value);
        }
        this.update(leaf_value);
    }

    get_value(c_puct) {
        // 获取价值，也就是论文中的Q+u，用来进行Select。
        // c_puct是参数，是UCT算法的变体，实际上控制开发和探索的权衡
        this._u = c_puct * this._P * Math.sqrt(this._parent._n_visits) / (1 + this._n_visits);
        return this._Q + this._u;
    }

    is_leaf() {
        // 判断是否是叶子节点，没有孩子节点的就是叶子结点。
        return Object.keys(this._children).length === 0;
    }

    is_root() {
        return !this._parent;
    }
}

class MCTS {
    // 实现蒙特卡洛树搜索

    constructor(policy_value_fn, c_puct = 5, n_playout = 1000) {
        // 初始化蒙特卡洛树搜索
        // 参数:
        //     policy_value_fn -- 是一个函数，输入棋盘状态，返回下一步的落子和概率，(action, probability)以及一个-1到1之间的分数。
        // 表示这一步导致的最后的胜负情况。
        // c_puct -- 0到正无穷之间的一个数，越大意味着越依赖以前
        this._root = new TreeNode(null, 1.0);
        this._policy = policy_value_fn;
        this._c_puct = c_puct;
        this._n_playout = n_playout;
    }

    _playout(game) {
        // 单次的蒙特卡洛搜索的模拟，即从根节点到叶节点一次，获得叶子节点价值然后反向传导，更新所有祖先节点
        let node = this._root;
        while (true) {
            if (node.is_leaf()) {
                break;
            }
            // 贪婪选择下一步
            // let {action, node} = node.select(this._c_puct)
            let res = node.select(this._c_puct);

            let action = res.action;
            node = res.node;

            game.do_move(action);  // 进行模拟落子一次
        }

        // 评估叶子结点，得到一系列的 (action, probability)
        // 以及一个-1到1之间的价值
        // state = game.state()
        let [action_probs, leaf_value] = this._policy(game);

        // 检查是否游戏结束
        let [game_over, winner] = game.has_a_winner();
        // 没有结束，扩展节点，利用网络输出的先验概率
        if (!game_over) {
            node.expand(action_probs);
        } else {
            // 结束了，返回真实的叶子结点值，不需要网络评估了。
            if (winner === -1) {
                leaf_value = 0.0;
            } else {
                if (winner === game.get_current_player()) {
                    leaf_value = 1.0;
                } else {
                    leaf_value = -1.0;
                }
            }
        }
        // 迭代更新所有祖先节点
        node.update_recursive(-leaf_value);
    }

    update_with_move(last_move) {
        if (last_move in this._root._children) {  // 根据对面的落子，复用子树，
            this._root = this._root._children[last_move];
            this._root._parent = null;
        } else {  // 否则，重新开始一个新的搜索树
            this._root = new TreeNode(null, 1.0);
        }
    }

    get_move_probs(game, temp = 1e-3) {
        /*多次模拟，并且根据子节点访问的次数和温度系数计算下一步落子的概率。
        温度系数0-1之间，控制探索的权重，越靠近1，分布越均匀，多样性大，
        越越接近0，分布越尖锐，追求最强棋力*/
        for (let n = 0; n < this._n_playout; n++) {
            let game_copy = _.cloneDeep(game);
            this._playout(game_copy);

        }
        // 根据访问次数计算落子概率
        let acts = Object.keys(this._root._children);
        let visits = Object.values(this._root._children).map((node) => {
            return node._n_visits;
        });
        let a = np.log(np.array(visits).add(1e-10)).multiply(1 / temp);
        let probs = softmax(a);
        return {acts, probs};
    }
}

export default class MTCSPlayer {

    // 初始化AI，基于MCTS
    constructor(policy_value_function, c_puct = 5, n_playout = 2000, is_selfplay = 0) {
        this.mcts = new MCTS(policy_value_function, c_puct, n_playout);
        this._is_selfplay = is_selfplay;
    }

    // 设置玩家
    set_player_ind(p) {
        this.player = p;
    }

    // 重置玩家
    reset_player() {
        this.mcts.update_with_move(-1);
    }

    // 获取落子
    get_action(board, temp = 1e-3, return_prob = 0) {
        let sensible_moves = board.availables;  // 获取所有可行的落子
        let move_probs = np.zeros(board.width * board.height);  // 获取落子的概率，由神经网络输出
        if (sensible_moves.length > 0) {
            // 棋盘未耗尽时
            let {acts, probs} = this.mcts.get_move_probs(board, temp);  // 获取落子以及对应的落子概率
            // todo
            // move_probs[...acts]= probs  // 将概率转到move_probs列表中
            acts.map((val, index) => {
                move_probs.set(val, probs.get(index));
            });

            let move;
            // 如果是自博弈的话
            if (this._is_selfplay) {
                // 为了增加探索，保证每个节点都有可能被选中，加入狄利克雷噪声
                // move = np.random.choice(acts,  0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs))));
                // let max = probs.max();
                // let i = probs.selection.data.indexOf(max);
                // move = acts[i];
                move = Random.choice(acts, probs.selection.data);

                this.mcts.update_with_move(move);  // 更新根节点，并且复用子树
            } else {
                // 如果采用默认的temp = le-3，几乎相当于选择最高概率的落子
                // move = np.random.choice(acts, probs);
                // 重置根节点
                let max = probs.max();
                let i = probs.selection.data.indexOf(max);
                move = acts[i];

                this.mcts.update_with_move(-1);
            }
            // 选择返回落子和相应的概率，还是只返回落子，因为自博弈时需要保存概率来训练网络，而真正落子时只要知道move就行了
            if (return_prob) {
                return {move, move_probs};
            } else {
                return move;
            }
        } else {
            console.error('WARNING: the board is full');
        }
    }

    toString() {
        return this.player || 'mcts_zero';
    }
}
