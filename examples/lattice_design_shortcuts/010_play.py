import xtrack as xt
env = xt.Environment()


env.new('drift.1', xt.Drift,      length='l.mq / 2')
env.new('qf',      xt.Quadrupole, k1='kqf.1', length='l.mq')
env.new('drift.2', xt.Replica,    parent_name='drift.1')
env.new('mb.1',    xt.Bend,       k0='k0.mb', h='k0.mb', length='l.mb')
env.new('mb.2',    xt.Replica,    parent_name='mb.1')
env.new('mb.3',    xt.Replica,    parent_name='mb.1')
env.new('drift.3', xt.Replica,    parent_name='drift.1')
env.new('qd',      xt.Quadrupole, k1='kqd.1', length='l.mq')
env.new('drift.4', xt.Replica,    parent_name='drift.1')

halfcell = env.new_line(components=[
    'drift.1',
    SPlace('qf', s
    'drift.2',
    'mb.1',
    'mb.2',
    'mb.3',
    'drift.3',
    'qd',
    'drift.4',
])

mbx = env.new('mbxw', xt.Bend, k0='k0.mb', h=0, length='l.mbxw')

d1 = env.new_line('d1', components=[
    env.new('lmbxw.start',   parent=xt.Marker), # shortcut env.new('lmbxw.start')
    env.new('mbxw.a4@start', parent=xt.Replica, parent_name='mbxw', at=0.5),
    env.new('mbxw.b4@start', parent='mbxw', from_='mbxw.a4@end'),
    env.new('lmbxw.end',     parent=xt.Marker, at=0.5, from_='mbxw.b4@end'),
])

d2 = env.new('d2.b1', xt.Bend, k0='k0.mb', h=0, length='l.mbxw', dx=0.188/2)

ir_left = env.new_line('ir_left', components=[
    env.new('ip1')
    Splace('d1r1@start', d1, at=100, from_='ip1'),
    Splace('d2@start', d2, at=200, from_='ip1'),
])

ir = evn.new_line(components=[
    ir_left.replicate('.l1', mirror=True),
    env.new('ip')
    ir_left.replicate('.r1')
])


s_ip1 = 0
s_ip2 = 2000
s_ip3 = 4000


lhc = env.new_line(components=[


    place('ir1', ir.replicate('1'), at=s_ip1, patch=True, reference=PatchReference('ip.1', x=0.1, xp=0.1)),
    place('ir2', ir.replicate('2'), at=s_ip2, patch=True, reference='ip.2'),


])

seq = [
    Place(env.new('ip', xt.Marker), at=10),
    Place(env.new('left', xt.Quadrupole, length=1), at=-5, from_='ip'),
    env.new('after_left', xt.Marker),
    env.new('after_left2', xt.Marker),
    Place(env.new('right',xt.Quadrupole, length=1), at=+5, from_='ip'),
    Place([env.new('before_right', xt.Quadrupole, length=1),
           env.new('before_right2', xt.Marker)], at=0, from_='right@start',
           anchor='before_right@center'),
    Place(env.new('righter', xt.Quadrupole, length=1), at=+5, from_='before_right'),
]