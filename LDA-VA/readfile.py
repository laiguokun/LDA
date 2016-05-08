class readfile:
	filename = "";
	doc = [];
	voca = {};
	doc_cnt = [];

	def __init__(self, name):
		self.filename = name;

	def read(self):
		file = open(self.filename);
		self.doc = [];
		self.voca = {};
		self.doc_cnt = [];
		cnt = 0;
		for line in file:
			d = [];
			d_cnt = [];
			words = line.strip().split();
			for i in range(1,len(words)):
				tmp = words[i].split(':');
				if (not tmp[0] in self.voca):
					self.voca[tmp[0]] = len(self.voca);
				d.append(int(self.voca[tmp[0]]));
				d_cnt.append(int(tmp[1]));
			self.doc.append(d);
			self.doc_cnt.append(d_cnt);
			cnt += len(d);
			if (len(self.doc) > 100):
				break;
		print('document length:' + str(cnt));
		print('documnet num:' + str(len(self.doc)));
		return self.doc, self.voca, self.doc_cnt;
	