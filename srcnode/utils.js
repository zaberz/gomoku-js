function zip() {
    var args = [].slice.call(arguments);
    var longest = args.reduce(function (a, b) {
        return a.length > b.length ? a : b;
    }, []);

    return longest.map(function (_, i) {
        return args.map(function (array) {
            return array[i];
        });
    });
}

function unzip() {

}

class Random {
    static choice(data, p) {
        if (data.length !== p.length) {
            throw new Error('error');
        }

        let total = Math.round(p.reduce((a, b) => (a + b)));

        let i = Math.random() * total;
        let count = 0;
        let index = 0;
        for (let v of p) {
            count += v;
            if (count > i) {
                break;
            }
            index++;
        }
        return data[index];

    }
}

module.exports = {
    zip,
    unzip,
    Random,
};
