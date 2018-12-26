import {Game, Board} from './game';
import PolicyValueNet from './policy_value_net_tensorflow';
import MCTSPlayer from './mcts_alphaZero';
import PurePlayer from './mcts_pure';

class Human {
    constructor(game) {
        this.player = null;
        this.game = game;
        this.UI = null;
    }

    set_player_ind(p) {
        this.player = p;
    }

    setUI(ui) {
        this.UI = ui;
    }

    get_action(board) {
        return new Promise((resolve, reject) => {
            this.UI.on('get_move', (move) => {
                console.log(move);
                if (board.availables.indexOf(move) > -1) {
                    resolve(move);
                }
            });

        });
    }

    toString() {
        return this.player || 'human';
    }
}

export default class Humanplay {
    constructor(size, n_in_row) {
        this.board = new Board(size, size, n_in_row);
        this.game = new Game(this.board);
        let best_model = `indexeddb://best_policy_model_${size}*${size}_${n_in_row}`;
        this.best_policy = new PolicyValueNet(size, size, best_model);
        this.mcts_player = new MCTSPlayer(this.best_policy.policy_value_fn.bind(this.best_policy), 5, 100);
        this.human_payer = new Human(this.game);
    }

    run() {
        this.game.start_human_play(this.human_payer, this.mcts_player, 0, true);
    }

    runWithPurePlayer() {
        const pure_player = new PurePlayer();
        this.game.start_human_play(this.human_payer, pure_player, 0, true);
    }
}

