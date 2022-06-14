from lifelike.constants import CHROMOSOME_LEN


def ones(rstring):
  ixsb = []
  ixss = []
  for i in range(CHROMOSOME_LEN // 2, 0, -1):
    if rstring & 1:
      ixsb.append(i)
    rstring >>= 1

  for i in range(CHROMOSOME_LEN // 2, -1, -1):
    if rstring & 1:
      ixss.append(i)
    rstring >>= 1
  return sorted(ixsb), sorted(ixss)