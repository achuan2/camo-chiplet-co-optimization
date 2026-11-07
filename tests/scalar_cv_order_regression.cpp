
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <vector>
struct Individual {
    double constraint_violation;
    std::vector<double> constraint_profile;
    std::string design_fingerprint;
    bool operator==(const Individual& rhs) const {
        return std::tie(constraint_violation,constraint_profile,design_fingerprint)
            == std::tie(rhs.constraint_violation,rhs.constraint_profile,rhs.design_fingerprint);
    }
};
Individual make(std::vector<double> profile, const std::string& id) {
    return {std::accumulate(profile.begin(),profile.end(),0.0),profile,id};
}
int main() {
    auto cmp = [](const Individual& lhs, const Individual& rhs) {
                    if (lhs.constraint_violation != rhs.constraint_violation)
                        return lhs.constraint_violation < rhs.constraint_violation;
                    if (lhs.constraint_profile != rhs.constraint_profile)
                        return lhs.constraint_profile < rhs.constraint_profile;
                    return lhs.design_fingerprint < rhs.design_fingerprint;
                };
    constexpr double e=1e-12;
    auto a=make({4*e,0,0,0,0,0},"A");
    auto b=make({3*e,1.75*e,0,0,0,0},"B");
    auto c=make({2*e,3.5*e,0,0,0,0},"C");
    if(cmp(a,c) && cmp(c,b) && !cmp(a,b)) {
        std::cout<<"FAIL: retained A<C, C<B, B<A comparator cycle\n";
        return 1;
    }
    std::vector<Individual> values={a,b,c,a,
        make({0,0,0,0,0,0},"zero"),
        make({2*e,2*e,0,0,0,0},"same-cv"),
        make({4*e,0,0,0,0,0},"A-second")};
    std::mt19937 rng(606);
    for(int n=0;n<64;++n) {
        std::vector<double> p(6);
        for(auto& v:p) v=(rng()%32)*e/4;
        values.push_back(make(p,std::to_string(n)));
    }
    std::size_t triples=0;
    for(const auto& x:values) {
        if(cmp(x,x)) return 2;
        for(const auto& y:values) {
            if(cmp(x,y) && cmp(y,x)) return 3;
            for(const auto& z:values) {
                ++triples;
                if(cmp(x,y) && cmp(y,z) && !cmp(x,z)) return 4;
                const auto eq=[&](const Individual& l,const Individual& r){return !cmp(l,r)&&!cmp(r,l);};
                if(eq(x,y) && eq(y,z) && !eq(x,z)) return 5;
            }
        }
    }
    auto reference=values;
    std::sort(reference.begin(),reference.end(),[](const Individual& l,const Individual& r){
        return std::tie(l.constraint_violation,l.constraint_profile,l.design_fingerprint)
             < std::tie(r.constraint_violation,r.constraint_profile,r.design_fingerprint);
    });
    for(int n=0;n<20;++n) {
        std::shuffle(values.begin(),values.end(),rng);
        std::stable_sort(values.begin(),values.end(),cmp);
        if(values!=reference) return 6;
    }
    std::cout<<"PASS: strict ordering, "<<triples<<" triples, 20 stable sorts\n";
}
