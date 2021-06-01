import minerl
import lz4.frame

data = minerl.data.make(
    'MineRLNavigateVectorObf-v0')
from PIL import Image

sizes = []
iter = 0
for current_state, action, reward, next_state, done in data.batch_iter(batch_size=1, num_epochs=1, seq_len=32):	
	#Image.fromarray(current_state["pov"][0][0]).save(f"im_{iter}.png")
	sizes.append(len(lz4.frame.compress(current_state["pov"][0][0])))
	iter += 1
print(sum(sizes)/len(sizes), 64*64*3)
