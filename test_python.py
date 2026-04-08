facts = {}
facts["bear_is_big"] = True
facts["bear_is_round"] = True
facts["bear_visits_cat"] = True
facts["bear_visits_dog"] = True
facts["cat_is_nice"] = True
facts["cat_likes_bear"] = True
facts["cat_likes_dog"] = True
facts["cat_visits_bear"] = True
facts["cat_visits_dog"] = True
facts["dog_is_nice"] = True
facts["dog_is_round"] = True
facts["dog_likes_bear"] = True
facts["dog_needs_bear"] = True
facts["dog_needs_cat"] = True
facts["dog_visits_bear"] = True
facts["dog_visits_cat"] = True

changed = True
while changed:
    changed = False
    
    if facts.get("bear_needs_dog") is True and facts.get("dog_visits_bear") is True:
        if facts.get("bear_likes_cat") is not True:
            facts["bear_likes_cat"] = True
            changed = True

if facts.get("dog_likes_dog") is True:
    print("RESULT: FALSE")
elif facts.get("dog_likes_dog") is False:
    print("RESULT: TRUE")
else:
    print("RESULT: UNCERTAIN")