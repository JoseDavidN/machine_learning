def viewMetrics(predictions, dataTest, name, viewAccuracy=False, viewPrecision=False, viewRecall=False, viewF1=False, viewCm=False, ax=None):
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
    import seaborn as sns
    
    accuracy = accuracy_score( dataTest, predictions[name] );
    precision = precision_score( dataTest, predictions[name], average="micro" );
    recall = recall_score( dataTest, predictions[name], average="micro" );
    f1 = f1_score( dataTest, predictions[name], average="micro" );
    cm = confusion_matrix( dataTest, predictions[name] );

    print(f"#--------- Resultados de {name} ---------#")
    resultsTitle = ''
    results = ''

    if viewAccuracy:
        resultsTitle += '-- Acc --\t'
        results += "  {0:.3f} \t".format( accuracy )
    if viewPrecision:
        resultsTitle += '-- Prec --\t'
        results += "  {0:.3f} \t".format( precision )
    if viewRecall:
        resultsTitle += '-- Rec --\t'
        results += "  {0:.3f} \t".format( recall )
    if viewF1:
        resultsTitle += '-- F1 --\t'
        results += "  {0:.3f} \t".format( f1 )
    if viewCm:
        sns.heatmap( cm, cmap='hot', annot=True, ax=ax );
        ax.set_title( name );

    print( resultsTitle )
    print( results )