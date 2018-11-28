import {Game, Board} from './game';
import PolicyValueNet from './policy_value_net_tensorflow';
import MCTSPlayer from './mcts_alphaZero';
import PUREMCTS from './mcts_pure';

class Human {
    constructor() {
        this.player = null;
    }

    set_player_ind(p) {
        this.player = p;
    }

    get_action(board) {
        return board.availables[0];
    }

    toString() {
        return this.player || 'human';
    }
}

function run(view) {
    let n = 3,
        width = 3,
        height = 3;

    let board = new Board(width, height, n);
    let game = new Game(board, view);

    // todo
    let best_model = 'indexeddb://best_policy_model'
    // let best_model = null;
    let best_policy = new PolicyValueNet(width, height, best_model);
    let mcts_player = new MCTSPlayer(best_policy.policy_value_fn.bind(best_policy), 5, 400);

    let pure_mcts = new PUREMCTS(4, 400);
    let human = new Human();
    // let player2 = pure_mcts;
    let player2 = mcts_player;
    // let player2 = new Human();
    game.start_play(human, player2, 0, 1);
}

let view = 0;
run(view);

