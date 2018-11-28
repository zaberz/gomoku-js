const _ = require('lodash');
const {zip} = require('./utils');

module.exports = class MCTSPlayer {
    constructor(c_puct = 5, n_playout = 2000) {
        this.mcts = new MCTS(policy_value_fn, c_puct, n_playout);
    }

    set_player_ind(p) {
        this.player = p;
    }

    reset_player() {
        this.mcts.update_with_move(-1);
    }

    get_action(board) {
        let sensible_moves = board.availables;
        if (sensible_moves.length > 0) {
            let move = this.mcts.get_move(board);
            this.mcts.update_with_move(-1);
            return move;
        } else {
            console.warn(`warning: the board is full`);
        }
    }

    toString() {
        return `MCTS ${this.player}`;
    }
};

class MCTS {
    constructor(policy_value_fn, c_puct = 5, n_play_out = 10000) {
        this._root = new TreeNode(null, 1.0);
        this._policy = policy_value_fn;
        this._c_puct = c_puct;
        this._n_playout = n_play_out;
    }

    _playout(state) {
        let node = this._root;
        while (true) {
            if (node.is_leaf()) {
                break;
            }
            let res = node.select(this._c_puct);
            let action = res.action;
            node = res.node;
            state.do_move(action);
        }

        let [action_probs, _] = this._policy(state);

        let [end, winner] = state.game_end();
        if (!end) {
            node.expand(action_probs);
        }
        let leaf_value = this._evaluate_rollout(state);
        node.update_recursive(-leaf_value);
    }

    _evaluate_rollout(state, limit = 1000) {
        let player = state.get_current_player();
        let end, winner;
        for (let i of _.range(limit)) {
            [end, winner] = state.game_end();
            if (end) {
                break;
            }
            let action_probs = rollout_policy_fn(state);
            let max_action = _.maxBy(action_probs, (act_prob) => {
                return act_prob[1];
            })[0];
            state.do_move(max_action);
        }
        if (!end) {
            console.warn(`warning: rollout reach move limit`);
        }
        if (winner === -1) {
            return 0;
        } else {
            if (winner === player) {
                return 1;
            } else {
                return -1;
            }
        }
    }

    get_move(state) {
        for (let n of _.range(this._n_playout)) {
            let state_copy = _.cloneDeep(state);
            this._playout(state_copy);
        }
        // let item = 0;
        // let max = 0;
        let children = this._root._children;

        let keys = Object.keys(children);

        let move = _.maxBy(keys, (key) => {
            return children[key]._n_visits;
        });
        // for (let key in children) {
        //     if (children[key]._n_visits > max) {
        //         max = children[key]._n_visits;
        //         item = key;
        //     }
        // }
        return move;
        // return item;
    }

    update_with_move(last_move) {
        if (this._root._children[last_move]) {
            this._root = this._root._children[last_move];
            this._root._parent = null;
        } else {
            this._root = new TreeNode(null, 1.0);
        }
    }
}

class TreeNode {
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

function rollout_policy_fn(board) {
    let action_probs = new Array(board.availables.length).fill(0).map(() => {
        return Math.random();
    });
    return zip(board.availables, action_probs);
}

function policy_value_fn(board) {
    let action_probs = new Array(board.availables.length).fill(1 / board.availables.length);
    return [zip(board.availables, action_probs), 0];
}