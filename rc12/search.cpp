// Blind stdin adapter for Wind's MIT-licensed ARC program synthesizer.
#include <iomanip>
#include "precompiled_stl.hpp"
using namespace std;
#include "utils.hpp"
#include "core_functions.hpp"
#include "image_functions.hpp"
#include "read.hpp"
#include "normalize.hpp"
#include "brute2.hpp"
#include "pieces.hpp"
#include "compose2.hpp"
#include "deduce_op.hpp"
#include "brute_size.hpp"

int MAXDEPTH, MAXSIDE = 100, MAXAREA = 1800, MAXPIXELS = 9000;
int print_times = 0, print_mem = 0, print_nodes = 0;

Image read_grid() {
  int h, w;
  if (!(cin >> h >> w) || min(h,w)<1 || max(h,w)>30) throw runtime_error("invalid shape");
  Image im = core::empty({w,h});
  for (auto &c : im.mask) {
    int v;
    if (!(cin >> v) || v<0 || v>9) throw runtime_error("invalid color");
    c = v;
  }
  return im;
}

int main(int argc,char** argv) {
  int n;
  if(argc!=3 || !(cin>>n) || n<1) return 2;
  MAXDEPTH = 10*stoi(argv[1]);
  const int mode = stoi(argv[2]);
  vector<pair<Image,Image>> data;
  for(int i=0;i<n;i++) {
    auto x=read_grid();
    data.emplace_back(x,read_grid());
  }
  auto query=read_grid();
  auto sim=mode==1 ? normalizeDummy(data) : normalizeCols(data);
  vector<pair<Image,Image>> train;
  int area=query.w*query.h;
  for(auto &[x,y]:data) {
    auto pair=sim(x,y);
    train.push_back(pair);
    area=max({area,x.w*x.h,y.w*y.h});
    if(mode>=2) train.emplace_back(rigid(pair.first,mode==2?6:7),rigid(pair.second,mode==2?6:7));
  }
  MAXAREA=area*2;
  MAXPIXELS=MAXAREA*5;
  auto x=sim.in(query);
  auto sizes=bruteSize(x,train);
  auto dags=brutePieces2(x,train,sizes);
  auto pieces=makePieces2(dags,train,sizes);
  for(auto &d:pieces.dag) {
    d.hashi.clear();
    for(auto &v:d.tiny_node.node) v.child.clear();
  }
  auto candidates=composePieces2(pieces,train,sizes);
  addDeduceOuterProduct(pieces,train,candidates);
  auto ranked=evaluateCands(candidates,train);
  set<vector<int>> seen;
  for(const auto &candidate:ranked) {
    auto img=sim.rec(query,candidate.imgs.back());
    if(min(img.w,img.h)<1 || max(img.w,img.h)>30) continue;
    vector<int> row={img.h,img.w};
    row.insert(row.end(),img.mask.begin(),img.mask.end());
    if(!seen.insert(row).second) continue;
    cout<<"GRID "<<setprecision(12)<<candidate.score<<" "<<train.size();
    for(auto v:row) cout<<" "<<v;
    cout<<"\n";
    if(seen.size()==8) break;
  }
}
