module.exports = class Player {
    set_player_ind(p) {
        this.player = p;
    }

    reset_player() {
    }

    get_action(board) {
        let sensible_moves = board.availables;
        if (sensible_moves.length > 0) {
            let move = sensible_moves[0];
            return move;
        } else {
            console.warn(`warning: the board is full`);
        }
    }

    toString() {
        return `SB Player ${this.player}`;
    }
};
