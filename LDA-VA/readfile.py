class readfile:
	filename = "";
	doc = {};
	voca = {};

	def __init__(self, name):
		self.filename = name;

	def read(self):
		file = open(self.filename);
		self.doc = [];
		self.voca = {};
		cnt = 0;
		for line in file:
			d = [];
			words = line.strip().split();
			for i in range(1,len(words)):
				tmp = words[i].split(':');
				if (not tmp[0] in self.voca):
					self.voca[tmp[0]] = len(self.voca);
				for j in range(int(tmp[1])):
					d.append(int(tmp[0]));
			self.doc.append(d);
			cnt += int(words[0]);
		print('document length:' + str(cnt));
		print('documnet num:' + str(len(self.doc)));
		return self.doc, self.voca;
	