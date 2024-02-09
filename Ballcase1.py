# Ballcase predictor casestudy using Decision Tree Classifier.


from sklearn import tree

def ML(weight,surface):
    BallFeatures = [[35,1],[47,1],[90,0],[48,1],[90,0],[35,1],[92,0],[35,1],[35,1],[35,1],[96,0]
        ,[43,1],[110,0],[35,1],[95,0]]
    Names = [1,1,2,1,2,1,2,1,1,1,2,1,2,1,2]
    clf = tree.DecisionTreeClassifier()
    clf.fit(BallFeatures,Names)
    result = clf.predict([[weight,surface]])

    if result==1:
        print("Your object looks like Tennis ball")
    elif result==2:
        print("Your object looks like Cricket ball")

def main():
    print("Enter a weight of object")
    weight=input()

    print("what is surface type of object Rough or Smooth")
    surface=input()
    if surface.lower() == "rough":
        surface = 1
    elif surface.lower == "smooth":
        surface = 0
    else :
        print("Error:Wrong input")
        exit()
    ML(weight,surface)

if __name__ == "__main__":
    main()
    

    
    
