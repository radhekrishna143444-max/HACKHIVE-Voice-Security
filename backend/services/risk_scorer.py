from config import LOW_THRESHOLD,HIGH_THRESHOLD
def assess(p,c):
 if p>=HIGH_THRESHOLD:return 'HIGH','Strong synthetic-voice indicators detected.','Alert + require additional verification'
 if p>=LOW_THRESHOLD:return 'MEDIUM','The analysis is uncertain or shows suspicious characteristics.','Require additional verification'
 return 'LOW','No strong synthetic-voice indicators were detected.','Allow, while maintaining normal security controls'
