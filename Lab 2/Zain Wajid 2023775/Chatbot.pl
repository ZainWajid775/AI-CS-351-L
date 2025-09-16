suspect(muqeem).
suspect(ashir).
suspect(salman).

location(muqeem, library).
location(ashir, library).
location(salman, cinema).

motive(muqeem, robbery).

weapon(muqeem, pistol).
weapon(ashir, rifle).

evidence(muqeem, fingerprint).

suspicious(X) :- location(X, library), motive(X,_).
dangerous(X) :- weapon(X,_).
murderer(X) :- suspicious(X), dangerous(X), evidence(X,_).