from time import sleep
from ttt_env import TTTEnv
from ttt_agents import *
import keyboard

agent = REINFORCE_MENTAgent(epsilon=0, learningRate=0, discountRate=0)
agent.load_weights("checkpoints/TTTREINFORCE_MENTAgent.tf")
env = TTTEnv(render_mode="human", opponent=agent)

keyboard.on_press_key('7', lambda _: (env.step(0)))
keyboard.on_press_key('8', lambda _: (env.step(1)))
keyboard.on_press_key('9', lambda _: (env.step(2)))
keyboard.on_press_key('4', lambda _: (env.step(3)))
keyboard.on_press_key('5', lambda _: (env.step(4)))
keyboard.on_press_key('6', lambda _: (env.step(5)))
keyboard.on_press_key('1', lambda _: (env.step(6)))
keyboard.on_press_key('2', lambda _: (env.step(7)))
keyboard.on_press_key('3', lambda _: (env.step(8)))
keyboard.on_press_key('esc', lambda _: exit())

#s, _ = env.reset()
#s = tf.expand_dims(tf.convert_to_tensor(s),0)
env.model.step(2.0, agent.act(tf.expand_dims(tf.convert_to_tensor(env.model.calcLogits(2.0)),0)))
while True:
    #s, _, _, _, _ = env.step(a)
    #s = tf.expand_dims(tf.convert_to_tensor(s),0)
    sleep(0.5)
    if env.model.terminated or env.model.truncated:
        sleep(3)
        s, _ = env.reset()
        s = tf.expand_dims(tf.convert_to_tensor(s),0)