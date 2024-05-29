

class HiddenVar:
    def __init__(self, son_list):
        self.son_list = son_list
        self.length = len(son_list)
        self.group = self.grouping()

    def grouping(self):
        group_list = []
        for variable in self.son_list:
            group_name = variable + '_group'
            group_list.append(group_name)

        return group_list


soilhm_son_list = ['cd', 'as', 'cr', 'hg', 'pb']
water_son_list = ['qtot', 'riverden']
air_son_list = ['pm10', 'pre', 'roaden', 'tmp', 'wsp']
farmer_son_list = ['gdp', 'iwu', 'pop']

soilhm_node = HiddenVar(soilhm_son_list)
water_node = HiddenVar(water_son_list)
air_node = HiddenVar(air_son_list)
farmer_node = HiddenVar(farmer_son_list)






