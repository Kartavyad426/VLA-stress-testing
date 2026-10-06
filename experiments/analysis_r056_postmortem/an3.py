import pickle,numpy as np
rows=pickle.load(open('rows.pkl','rb'))
def show(filt,title):
    print(title)
    for k,v in sorted(rows.items(),key=lambda x:str(x[0])):
        if not filt(k): continue
        s=' | '.join(f"{a}:{'S' if v[a]['success'] else 'F'} st{v[a]['steps']:3d} lift_t {str(v[a]['lift_t']):>4} maxlift {v[a]['maxlift']*100:5.1f}cm dmin {v[a]['dmin']*100:4.1f}" for a in ['base','r16','r4'])
        print(k[1:],s)
show(lambda k:k[1]==1327,'scene 1327')
