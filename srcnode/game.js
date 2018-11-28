const nj = require('numjs');
const _ = require('lodash');
const EventEmitter = require('eventemitter3');
const {zip} = require('./utils');

class Board {
    constructor(width = 8, height = 8, n_in_row = 5) {
        this.width = width;
        this.height = height;
        this.states = {};
        this.n_in_row = n_in_row;
        this.players = [1, 2];
    }

    init_board(start_play = 0) {
        if (this.width < this.n_in_row || this.height < this.n_in_row) {
            throw new Error(`board width and height can not be less than ${this.n_in_row}`);
        }

        this.current_player = this.players[start_play];
        this.availables = _.range(this.width * this.height);
        this.states = {};
        this.last_move = -1;
    }

    move_to_location(move) {
        /*3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)*/

        let h = Math.floor(move / this.width);
        let w = move % this.height;
        return [h, w];
    }

    location_to_move(location) {
        if (location.length !== 2) {
            return -1;
        }
        let [h, w] = location;
        let move = h * this.width + w;
        if (move < this.width * this.height) {
            return -1;
        }
        return move;
    }

    current_state() {
        let square_state = nj.zeros([4, this.width, this.height]);

        if (this.states) {
            let moves = Object.keys(this.states);
            let move_curr = moves.filter(item => {
                return this.states[item] === this.current_player;
            });

            let move_oppo = moves.filter(item => {
                return this.states[item] !== this.current_player;
            });

            move_curr.map(i => {
                square_state.set(0, Math.floor(i / this.width), i % this.height, 1.0);
            });

            move_oppo.map(i => {
                square_state.set(1, Math.floor(i / this.width), i % this.height, 1.0);
            });
            this.last_move > -1 && square_state.set(2, Math.floor(this.last_move / this.width), this.last_move % this.height, 1.0);

            if (moves.length % 2 === 0) {
                for (let i = 0; i < this.width; i++) {
                    for (let j = 0; j < this.height; j++) {
                        square_state.set(3, i, j, 1.0);
                    }
                }
            }
        }
        return square_state;
    }

    do_move(move) {
        move = parseInt(move);
        this.states[move] = this.current_player;

        _.pull(this.availables, move);

        if (this.current_player === this.players[1]) {
            this.current_player = this.players[0];
        } else {
            this.current_player = this.players[1];
        }
        this.last_move = move;
    }

    has_a_winner() {
        let {width, height, states} = this;
        let n = this.n_in_row;
        let moved = Object.keys(states);

        if (moved.length < this.n_in_row + 2) {
            return [false, -1];
        }

        for (let m of moved) {
            m = parseInt(m);
            let h = Math.floor(m / width);
            let w = m % width;
            let player = states[m];
            if (w < width - n + 1) {
                // 优化减少遍历
                // 横向
                let arr = [];
                for (let i of _.range(m, m + n)) {
                    arr.push(states[i]);
                }
                if ((new Set(arr)).size === 1) {
                    return [true, player];
                }
            }

            if (h < height - n + 1) {
                let arr = [];
                for (let i of _.range(m, m + n * width, width)) {
                    arr.push(states[i]);
                }
                if ((new Set(arr)).size === 1) {
                    return [true, player];
                }
            }

            if (w < width - n + 1 && h < height - n + 1) {
                let arr = [];
                for (let i of _.range(m, m + n * (width + 1), width + 1)) {

                    arr.push(states[i]);
                }
                if ((new Set(arr)).size === 1) {
                    return [true, player];
                }
            }

            if (w >= n - 1 && w <= width && h < height - n + 1) {
                let arr = [];
                for (let i of _.range(m, m + n * (width - 1), width - 1)) {

                    arr.push(states[i]);
                }
                if ((new Set(arr)).size === 1) {
                    return [true, player];
                }
            }
        }
        return [false, -1];
    }

    game_end() {
        let [win, winner] = this.has_a_winner();
        if (win) {
            return [true, winner];
        } else if (!this.availables.length) {
            return [true, -1];
        }
        return [false, -1];
    }

    get_current_player() {
        return this.current_player;
    }

}

class Game extends EventEmitter {
    constructor(board) {
        super();
        this.board = board;

    }

    graphic(board, player1, player2) {
        let width = board.width;
        let height = board.height;
        console.group();
        console.log(`Player${player1},with X`);
        console.log(`Player${player1},with O`);
        let a = _.range(width);
        a.unshift(' ');
        console.log(a.join(' '));

        for (let i of _.range(height - 1, -1, -1)) {
            let arr = [];
            arr.push(i);
            for (let j of _.range(width)) {
                let loc = i * width + j;
                let p = board.states[loc];
                if (p === player1) {
                    arr.push('X');
                } else if (p === player2) {
                    arr.push('O');
                } else {
                    arr.push('-');
                }
            }
            console.log(arr.join(' '));
        }
        console.groupEnd();
    }

    start_play(player1, player2, start_player = 0, is_shown = 1) {
        if ([0, 1].indexOf(start_player) === -1) {
            throw new Error(`start_player should be either 0 (player1 first) or 1 (player2 first)`);
        }
        this.board.init_board(start_player);
        let [p1, p2] = this.board.players;
        player1.set_player_ind(p1);
        player2.set_player_ind(p2);

        let players = {1: player1, 2: player2};

        if (is_shown) {
            this.graphic(this.board, player1.player, player2.player);
        }

        while (true) {
            let current_player = this.board.get_current_player();
            let player_in_turn = players[current_player];
            let move = player_in_turn.get_action(this.board);
            this.board.do_move(move);
            if (is_shown) {
                this.graphic(this.board, player1.player, player2.player);
            }
            let [end, winner] = this.board.game_end();
            if (end) {
                if (is_shown) {
                    if (winner !== -1) {
                        console.log(`Game end. Winner is ${players[winner]}`);
                    } else {
                        console.log(`Game end. Tie`);
                    }
                }
                return winner;
            }
        }
    }

    async start_self_play(player, temp = 1e-3, is_shown = 0) {
        this.board.init_board();
        let [p1, p2] = this.board.players;
        let states = [],
            mcts_probs = [],
            current_players = [];
        while (1) {
            let {move, move_probs} = player.get_action(this.board, temp, 1);
            states.push(this.board.current_state(true));
            mcts_probs.push(move_probs);
            current_players.push(this.board.current_player);

            this.board.do_move(move);

            if (is_shown) {
                // todo show graphic
                this.graphic(this.board, p1, p2);
            }

            let [end, winner] = this.board.game_end();
            if (end) {
                let winners_z = new Array(current_players.length).fill(0);
                if (winner !== -1) {
                    current_players.map((val, index) => {
                        if (val === winner) {
                            winners_z[index] = 1.0;
                        } else {
                            winners_z[index] = -1.0;
                        }

                    });
                }
                player.reset_player();
                if (is_shown) {
                    if (winner !== -1) {
                        console.log(`Game end. Winner is player: ${winner}`);
                    } else {
                        console.log(`Game end.Tie`);
                    }
                }
                return [winner, zip(states, mcts_probs, winners_z)];
            }
        }
    }
}

module.exports = {
    Board,
    Game,
};