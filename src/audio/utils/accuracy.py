import numpy as np
from sklearn import metrics
    

class MeanWeightedAccuracyMeasure:
    """Emotional multilabel mWA
    
    Args:
        name (str, optional): Performance measure name
    """
    def __init__(self, name: str = 'mWA') -> None:
        self.name = name
        self.wa_scores = []
        
    def cmu_accuracy(self, 
                     targets: list[np.ndarray] | np.ndarray, 
                     predicts: list[np.ndarray] | np.ndarray) -> float:
        """Calculates CMU accuracy using formula: 
        A = (TP * N / P + TN) / (2 * N)

        Args:
            targets (list[np.ndarray] | np.ndarray): Targets array
            predicts (list[np.ndarray] | np.ndarray): Predicts array

        Returns:
            float: CMU Acc Value
        """
        true_label = (targets > 0)
        predicted_label = (predicts > 0)
        tp = float(np.sum((true_label==1) & (predicted_label==1)))
        tn = float(np.sum((true_label==0) & (predicted_label==0)))
        p = float(np.sum(true_label==1))
        n = float(np.sum(true_label==0))
        
        return (tp * (n / (p + 1e-16)) + tn) / (2 * n + 1e-16)
           
    def calc(self, targets: list[np.ndarray] | np.ndarray, predicts: list[np.ndarray] | np.ndarray) -> float:
        """Calculates mean Weighted Accuracy (based on CMU Acc) 

        Args:
            targets (list[np.ndarray] | np.ndarray): Targets array
            predicts (list[np.ndarray] | np.ndarray): Predicts array

        Returns:
            float: mWA value
        """
        targets = np.array(targets)
        predicts = np.array(predicts)
        self.wa_scores = []
        for i in range(0, predicts.shape[1]):
            self.wa_scores.append(self.cmu_accuracy(targets[:, i], predicts[:, i]) * 100)
        
        return np.mean(self.wa_scores)
    
    def get_scores(self) -> list[float]:
        """Get class-wise measures

        Returns:
            list[float]: Class-wise measure values
        """
        return self.wa_scores        
    
    def __str__(self) -> str:
        """Get name of performance measure

        Returns:
            str: Name of measure
        """
        return self.name
    

class MeanWeightedF1Measure:
    """Emotional multilabel mWF1
    
    Args:
        name (str, optional): Performance measure name
    """
    def __init__(self, name: str = 'mWF1') -> None:
        self.name = name
        self.f1_weighted_scores = []
    
    def calc(self, targets: list[np.ndarray] | np.ndarray, predicts: list[np.ndarray] | np.ndarray) -> float:
        """Calculates mean Weighted F1

        Args:
            targets (list[np.ndarray] | np.ndarray): Targets array
            predicts (list[np.ndarray] | np.ndarray): Predicts array

        Returns:
            float: mWF1 value
        """
        targets = np.array(targets)
        predicts = np.array(predicts)

        self.f1_weighted_scores = []
        for i in range(0, predicts.shape[1]):
            cr = metrics.classification_report(targets[:, i], predicts[:, i], output_dict=True)
            self.f1_weighted_scores.append(cr['weighted avg']['f1-score'] * 100)

        return np.mean(self.f1_weighted_scores)
    
    def get_scores(self) -> list[float]:
        """Get class-wise measures

        Returns:
            list[float]: Class-wise measure values
        """
        return self.f1_weighted_scores  
    
    def __str__(self) -> str:
        """Get name of performance measure

        Returns:
            str: Name of measure
        """
        return self.name


class MeanMacroF1Measure:
    """Emotional multilabel mMacroF1
    
    Args:
        name (str, optional): Performance measure name
    """
    def __init__(self, name: str = 'mMacroF1') -> None:
        self.name = name
        self.f1_macro_scores = []
    
    def calc(self, targets: list[np.ndarray] | np.ndarray, predicts: list[np.ndarray] | np.ndarray) -> float:
        """Calculates mean Macro F1

        Args:
            targets (list[np.ndarray] | np.ndarray): Targets array
            predicts (list[np.ndarray] | np.ndarray): Predicts array

        Returns:
            float: mMacroF1 value
        """
        targets = np.array(targets)
        predicts = np.array(predicts)

        self.f1_macro_scores = []
        for i in range(0, predicts.shape[1]):
            cr = metrics.classification_report(targets[:, i], predicts[:, i], output_dict=True)
            self.f1_macro_scores.append(cr['macro avg']['f1-score'] * 100)

        return np.mean(self.f1_macro_scores)
    
    def get_scores(self) -> list[float]:
        """Get class-wise measures

        Returns:
            list[float]: Class-wise measure values
        """
        return self.f1_macro_scores  
    
    def __str__(self) -> str:
        """Get name of performance measure

        Returns:
            str: Name of measure
        """
        return self.name


class MeanUARMeasure:
    """Emotional multilabel mUAR
    
    Args:
        name (str, optional): Performance measure name
    """
    def __init__(self, name: str = 'mUAR') -> None:
        self.name = name
        self.uar = []
    
    def calc(self, targets: list[np.ndarray] | np.ndarray, predicts: list[np.ndarray] | np.ndarray) -> float:
        """Calculates mean Unweighted Average Recall

        Args:
            targets (list[np.ndarray] | np.ndarray): Targets array
            predicts (list[np.ndarray] | np.ndarray): Predicts array

        Returns:
            float: mUAR value
        """
        targets = np.array(targets)
        predicts = np.array(predicts)

        self.uar = []
        for i in range(0, predicts.shape[1]):
            cr = metrics.classification_report(targets[:, i], predicts[:, i], output_dict=True, zero_division=0)
            self.uar.append(cr['macro avg']['recall'] * 100)
            
        return np.mean(self.uar)
    
    def get_scores(self) -> list[float]:
        """Get class-wise measures

        Returns:
            list[float]: Class-wise measure values
        """
        return self.uar  
    
    def __str__(self) -> str:
        """Get name of performance measure

        Returns:
            str: Name of measure
        """
        return self.name
    
    
class WARMeasure: 
    """A(WAR) aka accuracy
    Used for sentiment recognition in CMUMOSEI or 
    sentiment recognition in MELD/RAMAS or
    emotion recognition in MELD/RAMAS
    
    Args:
        name (str, optional): Performance measure name
    """
    def __init__(self, name: str = 'A(WAR)') -> None:
        self.name = name
        pass
    
    def calc(self, targets: list[np.ndarray] | np.ndarray, predicts: list[np.ndarray] | np.ndarray) -> float:
        """Calculates Accuracy or Weighted Average Recall

        Args:
            targets (list[np.ndarray] | np.ndarray): Targets array
            predicts (list[np.ndarray] | np.ndarray): Predicts array

        Returns:
            float: A(WAR) value
        """
        targets = np.array(targets)
        predicts = np.array(predicts)
        cr = metrics.classification_report(targets, predicts, output_dict=True, zero_division=0)
        return cr['weighted avg']['recall'] * 100
    
    def __str__(self) -> str:
        """Get name of performance measure

        Returns:
            str: Name of measure
        """
        return self.name
    

class UARMeasure: 
    """UAR
    Used for sentiment recognition in CMUMOSEI or 
    sentiment recognition in MELD/RAMAS or
    emotion recognition in MELD/RAMAS
    
    Args:
        name (str, optional): Performance measure name
    """
    def __init__(self, name: str = 'UAR') -> None:
        self.name = name
        pass
    
    def calc(self, targets: list[np.ndarray] | np.ndarray, predicts: list[np.ndarray] | np.ndarray) -> float:
        """Calculates Unweighted Average Recall

        Args:
            targets (list[np.ndarray] | np.ndarray): Targets array
            predicts (list[np.ndarray] | np.ndarray): Predicts array

        Returns:
            float: UAR value
        """
        targets = np.array(targets)
        predicts = np.array(predicts)
        cr = metrics.classification_report(targets, predicts, output_dict=True, zero_division=0)
        return cr['macro avg']['recall'] * 100
    
    def __str__(self) -> str:
        """Get name of performance measure

        Returns:
            str: Name of measure
        """
        return self.name


class WeightedF1Measure:
    """WF1
    Used for sentiment recognition in CMUMOSEI or 
    sentiment recognition in MELD/RAMAS or
    emotion recognition in MELD/RAMAS
    
    Args:
        name (str, optional): Performance measure name
    """
    def __init__(self, name: str = 'WF1') -> None:
        self.name = name
        pass
    
    def calc(self, targets: list[np.ndarray] | np.ndarray, predicts: list[np.ndarray] | np.ndarray) -> float:
        """Calculates Weighted F1

        Args:
            targets (list[np.ndarray] | np.ndarray): Targets array
            predicts (list[np.ndarray] | np.ndarray): Predicts array

        Returns:
            float: WF1 value
        """
        targets = np.array(targets)
        predicts = np.array(predicts)
        cr = metrics.classification_report(targets, predicts, output_dict=True, zero_division=0)
        return cr['weighted avg']['f1-score'] * 100
    
    def __str__(self) -> str:
        """Get name of performance measure

        Returns:
            str: Name of measure
        """
        return self.name
    
    
class MacroF1Measure: 
    """MacroF1
    Used for sentiment recognition in CMUMOSEI or 
    sentiment recognition in MELD/RAMAS or
    emotion recognition in MELD/RAMAS
    
    Args:
        name (str, optional): Performance measure name
    """
    def __init__(self, name: str = 'MacroF1') -> None:
        self.name = name
        pass
    
    def calc(self, targets: list[np.ndarray] | np.ndarray, predicts: list[np.ndarray] | np.ndarray) -> float:
        """Calculates Macro F1

        Args:
            targets (list[np.ndarray] | np.ndarray): Targets array
            predicts (list[np.ndarray] | np.ndarray): Predicts array

        Returns:
            float: MacroF1 value
        """
        targets = np.array(targets)
        predicts = np.array(predicts)
        cr = metrics.classification_report(targets, predicts, output_dict=True, zero_division=0)
        return cr['macro avg']['f1-score'] * 100
    
    def __str__(self) -> str:
        """Get name of performance measure

        Returns:
            str: Name of measure
        """
        return self.name


class MultilabelEmoSenCombinedMeasure:
    """Combined multilabel measure: mean of multilabel Emotional meanMacroF1 and Sentiment MacroF1
    
    Args:
        name (str, optional): Performance measure name
    """
    def __init__(self, name: str = 'MultilabelEmoSenCombined') -> None:
        self.name = name
        self.emo_m_macro_f1 = MeanMacroF1Measure()
        self.sen_macro_f1 = MacroF1Measure()
           
    def calc(self, targets: dict[list[np.ndarray]] | dict[np.ndarray], predicts: dict[list[np.ndarray]] | dict[np.ndarray]) -> float:
        """Calculates mean of multilabel Emotional meanMacroF1 and Sentiment MacroF1

        Args:
            targets (dict[list[np.ndarray]] | dict[np.ndarray]): Targets array
            predicts (dict[list[np.ndarray]] | dict[np.ndarray]): Predicts array

        Returns:
            float: mean of meanMacroF1 and MacroF1
        """
        emo_performance = self.emo_m_macro_f1.calc(targets['emo'], predicts['emo'])
        sen_performance = self.sen_macro_f1.calc(targets['sen'], predicts['sen'])
        return (emo_performance + sen_performance) / 2      
    
    def __str__(self) -> str:
        """Get name of performance measure

        Returns:
            str: Name of measure
        """
        return self.name
    
    
class EmoSenCombinedMeasure:
    """Combined measure: Emotional and Sentiment MacroF1
    
    Args:
        name (str, optional): Performance measure name
    """
    def __init__(self, name: str = 'EmoSenCombined') -> None:
        self.name = name
        self.emo_macro_f1 = MacroF1Measure()
        self.sen_macro_f1 = MacroF1Measure()
           
    def calc(self, targets: dict[list[np.ndarray]] | dict[np.ndarray], predicts: dict[list[np.ndarray]] | dict[np.ndarray]) -> float:
        """Calculates mean of Emotional and Sentiment MacroF1

        Args:
            targets (dict[list[np.ndarray]] | dict[np.ndarray]): Targets array
            predicts (dict[list[np.ndarray]] | dict[np.ndarray]): Predicts array

        Returns:
            float: mean of two MacroF1
        """
        targets = np.array(targets)
        predicts = np.array(predicts)

        emo_performance = self.emo_macro_f1.calc(targets['emo'], predicts['emo'])
        sen_performance = self.sen_macro_f1.calc(targets['sen'], predicts['sen'])
        return (emo_performance + sen_performance) / 2    
    
    def __str__(self) -> str:
        """Get name of performance measure

        Returns:
            str: Name of measure
        """
        return self.name
    

if __name__ == "__main__":
    m = UARMeasure()
    print(m)
    print('{0} = {1:.3f}%'.format(m, m.calc([1, 2, 3], [3, 2, 3])))