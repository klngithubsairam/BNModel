import re


txt = "pragma solidity function airDrop() require(keccak256) send(1011) while(i<200){ .. } \
        require(tokenBalance[msg.sender]) function supportsToken()"
        

pattern1 = r'function [^()]*\(\)'    # function *()  //pattern-1
result1 = re.findall(pattern1, txt)
print(result1, "Probability=", len(result1)/6)

pattern2 = r'require\([^)]*\)'    # require(*)   // pattern-9
result2 = re.findall(pattern2, txt)
print(result2, "Probability=", len(result2)/6)

pattern3 = r'send\(\d+\)'       # send(digits)     //pattern-13
result3 = re.findall(pattern3, txt)
print(result3, "Probability=", len(result3)/6)

pattern4 = r'while\([^)]*\)\{'      # while(condition){      //pattern-8
result4 = re.findall(pattern4, txt)
print(result4, "Probability=", len(result4)/6)
