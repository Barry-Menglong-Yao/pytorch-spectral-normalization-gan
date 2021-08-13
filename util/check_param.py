

#https://gist.github.com/ptrblck/d9abccd4f52b1aa6d242da3338533169
class ParamChecker():
    def __init__(self,model):
        self.save_init_values(model)

    def save_init_values(self,model):
    
        self.old_state_dict = {}
        for key in model.state_dict():
            self.old_state_dict[key] = model.state_dict()[key].clone()


    def print_grad_after_backward(self,model):
        params = model.state_dict()
        for name, grads  in params.items():
            print(f'{name} grad: {grads.grad} ')
        

    def save_new_params(self,model):
        self.new_state_dict = {}
        for key in model.state_dict():
            self.new_state_dict[key] = model.state_dict()[key].clone()
    
   

    def compare_params(self,model):
        self.save_new_params(model)
        for key in self.old_state_dict:
            if not (self.old_state_dict[key] == self.new_state_dict[key]).all():
                print('Diff in {}'.format(key))

        self.old_state_dict=self.new_state_dict


    def update(self,model):
        self.old_state_dict = {}
        for key in model.state_dict():
            self.old_state_dict[key] = model.state_dict()[key].clone()



def print_params(state_dict) :
    for key,value in  state_dict.items() :
        print(f"{key}={value}")