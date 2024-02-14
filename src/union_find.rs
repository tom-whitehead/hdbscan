pub(crate) struct UnionFind {
    parent: Vec<usize>,
    next_label: usize,
    size: Vec<usize>,
}

impl UnionFind {

    pub(crate) fn new(n_samples: usize) -> Self {
        let length = 2 * n_samples - 1;
        let parent = vec![length; length];
        let next_label = n_samples;
        let size = (0..length).into_iter().map(|n| {
            if n < n_samples { 1 } else { 0 }
        }).collect();

        UnionFind { parent, next_label, size }
    }

    pub(crate) fn union(&mut self, m: usize, n: usize) {
        self.parent[m] = self.next_label;
        self.parent[n] = self.next_label;
        self.size[self.next_label] = self.size[m] + self.size[n];
        self.next_label += 1;
    }

    pub(crate) fn find(&mut self, mut n: usize) -> usize {
        let mut p = n;
        while self.parent[n] != self.parent.len() {
            n = self.parent[n];
        }
        while self.parent[p] != n {
            p = self.wrap_parent_index_if_necessary(p);
            p = self.parent[p];
            p = self.wrap_parent_index_if_necessary(p);
            self.parent[p] = n;
        }
        n
    }

    pub(crate) fn size_of(&self, n: usize) -> usize {
        self.size[n]
    }

    fn wrap_parent_index_if_necessary(&self, idx: usize) -> usize {
        if idx == self.parent.len() {
            self.parent.len() - 1
        } else {
            idx
        }
    }
}
