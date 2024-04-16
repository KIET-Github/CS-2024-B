class HealthProfile {
  int? _age;
  int? _height;
  double? _weight;
  double? _avgHoursOfSleep;
  double? _avgActivityHours;
  String? _gender;
  List<String>? _medicalConditions;
  List<String>? _allergies;
  String? _diet;
  List<String>? _exerciseRoutine;
  String? _smokingHabits;
  String? _alcoholConsumption;
  String? _recreationalDrugUse;
  String? _stressManagement;
  List<String>? _mentalHealthHistory;
  List<String>? _familyHistory;
  String? _sleepSchedule;
  String? _screenTime;
  String? _occupation;
  String? _occupationalExposures;
  String? _travelHistory;
  String? _menstrualHistory;
  String? _additionalNotes;

  HealthProfile(
      {int? age,
      int? height,
      double? weight,
      double? avgHoursOfSleep,
      double? avgActivityHours,
      String? gender,
      List<String>? medicalConditions,
      List<String>? allergies,
      String? diet,
      List<String>? exerciseRoutine,
      String? smokingHabits,
      String? alcoholConsumption,
      String? recreationalDrugUse,
      String? stressManagement,
      List<String>? mentalHealthHistory,
      List<String>? familyHistory,
      String? sleepSchedule,
      String? screenTime,
      String? occupation,
      String? occupationalExposures,
      String? travelHistory,
      String? menstrualHistory,
      String? additionalNotes}) {
    if (age != null) {
      this._age = age;
    }
    if (height != null) {
      this._height = height;
    }
    if (weight != null) {
      this._weight = weight;
    }
    if (avgHoursOfSleep != null) {
      this._avgHoursOfSleep = avgHoursOfSleep;
    }
    if (avgActivityHours != null) {
      this._avgActivityHours = avgActivityHours;
    }
    if (gender != null) {
      this._gender = gender;
    }
    if (medicalConditions != null) {
      this._medicalConditions = medicalConditions;
    }
    if (allergies != null) {
      this._allergies = allergies;
    }
    if (diet != null) {
      this._diet = diet;
    }
    if (exerciseRoutine != null) {
      this._exerciseRoutine = exerciseRoutine;
    }
    if (smokingHabits != null) {
      this._smokingHabits = smokingHabits;
    }
    if (alcoholConsumption != null) {
      this._alcoholConsumption = alcoholConsumption;
    }
    if (recreationalDrugUse != null) {
      this._recreationalDrugUse = recreationalDrugUse;
    }
    if (stressManagement != null) {
      this._stressManagement = stressManagement;
    }
    if (mentalHealthHistory != null) {
      this._mentalHealthHistory = mentalHealthHistory;
    }
    if (familyHistory != null) {
      this._familyHistory = familyHistory;
    }
    if (sleepSchedule != null) {
      this._sleepSchedule = sleepSchedule;
    }
    if (screenTime != null) {
      this._screenTime = screenTime;
    }
    if (occupation != null) {
      this._occupation = occupation;
    }
    if (occupationalExposures != null) {
      this._occupationalExposures = occupationalExposures;
    }
    if (travelHistory != null) {
      this._travelHistory = travelHistory;
    }
    if (menstrualHistory != null) {
      this._menstrualHistory = menstrualHistory;
    }
    if (additionalNotes != null) {
      this._additionalNotes = additionalNotes;
    }
  }

  int? get age => _age;
  set age(int? age) => _age = age;
  int? get height => _height;
  set height(int? height) => _height = height;
  double? get weight => _weight;
  set weight(double? weight) => _weight = weight;
  double? get avgHoursOfSleep => _avgHoursOfSleep;
  set avgHoursOfSleep(double? avgHoursOfSleep) =>
      _avgHoursOfSleep = avgHoursOfSleep;
  double? get avgActivityHours => _avgActivityHours;
  set avgActivityHours(double? avgActivityHours) =>
      _avgActivityHours = avgActivityHours;
  String? get gender => _gender;
  set gender(String? gender) => _gender = gender;
  List<String>? get medicalConditions => _medicalConditions;
  set medicalConditions(List<String>? medicalConditions) =>
      _medicalConditions = medicalConditions;
  List<String>? get allergies => _allergies;
  set allergies(List<String>? allergies) => _allergies = allergies;
  String? get diet => _diet;
  set diet(String? diet) => _diet = diet;
  List<String>? get exerciseRoutine => _exerciseRoutine;
  set exerciseRoutine(List<String>? exerciseRoutine) =>
      _exerciseRoutine = exerciseRoutine;
  String? get smokingHabits => _smokingHabits;
  set smokingHabits(String? smokingHabits) => _smokingHabits = smokingHabits;
  String? get alcoholConsumption => _alcoholConsumption;
  set alcoholConsumption(String? alcoholConsumption) =>
      _alcoholConsumption = alcoholConsumption;
  String? get recreationalDrugUse => _recreationalDrugUse;
  set recreationalDrugUse(String? recreationalDrugUse) =>
      _recreationalDrugUse = recreationalDrugUse;
  String? get stressManagement => _stressManagement;
  set stressManagement(String? stressManagement) =>
      _stressManagement = stressManagement;
  List<String>? get mentalHealthHistory => _mentalHealthHistory;
  set mentalHealthHistory(List<String>? mentalHealthHistory) =>
      _mentalHealthHistory = mentalHealthHistory;
  List<String>? get familyHistory => _familyHistory;
  set familyHistory(List<String>? familyHistory) =>
      _familyHistory = familyHistory;
  String? get sleepSchedule => _sleepSchedule;
  set sleepSchedule(String? sleepSchedule) => _sleepSchedule = sleepSchedule;
  String? get screenTime => _screenTime;
  set screenTime(String? screenTime) => _screenTime = screenTime;
  String? get occupation => _occupation;
  set occupation(String? occupation) => _occupation = occupation;
  String? get occupationalExposures => _occupationalExposures;
  set occupationalExposures(String? occupationalExposures) =>
      _occupationalExposures = occupationalExposures;
  String? get travelHistory => _travelHistory;
  set travelHistory(String? travelHistory) => _travelHistory = travelHistory;
  String? get menstrualHistory => _menstrualHistory;
  set menstrualHistory(String? menstrualHistory) =>
      _menstrualHistory = menstrualHistory;
  String? get additionalNotes => _additionalNotes;
  set additionalNotes(String? additionalNotes) =>
      _additionalNotes = additionalNotes;

  HealthProfile.fromJson(Map<String, dynamic> json) {
    _age = json['age'];
    _height = json['height'];
    _weight = json['weight'];
    _avgHoursOfSleep = json['avgHoursOfSleep'];
    _avgActivityHours = json['avgActivityHours'];
    _gender = json['gender'];
    _medicalConditions = json['medicalConditions'].cast<String>();
    _allergies = json['allergies'].cast<String>();
    _diet = json['diet'];
    _exerciseRoutine = json['exerciseRoutine'].cast<String>();
    _smokingHabits = json['smokingHabits'];
    _alcoholConsumption = json['alcoholConsumption'];
    _recreationalDrugUse = json['recreationalDrugUse'];
    _stressManagement = json['stressManagement'];
    _mentalHealthHistory = json['mentalHealthHistory'].cast<String>();
    _familyHistory = json['familyHistory'].cast<String>();
    _sleepSchedule = json['sleepSchedule'];
    _screenTime = json['screenTime'];
    _occupation = json['occupation'];
    _occupationalExposures = json['occupationalExposures'];
    _travelHistory = json['travelHistory'];
    _menstrualHistory = json['menstrualHistory'];
    _additionalNotes = json['additionalNotes'];
  }

  Map<String, dynamic> toJson() {
    final Map<String, dynamic> data = new Map<String, dynamic>();
    data['age'] = this._age;
    data['height'] = this._height;
    data['weight'] = this._weight;
    data['avgHoursOfSleep'] = this._avgHoursOfSleep;
    data['avgActivityHours'] = this._avgActivityHours;
    data['gender'] = this._gender;
    data['medicalConditions'] = this._medicalConditions;
    data['allergies'] = this._allergies;
    data['diet'] = this._diet;
    data['exerciseRoutine'] = this._exerciseRoutine;
    data['smokingHabits'] = this._smokingHabits;
    data['alcoholConsumption'] = this._alcoholConsumption;
    data['recreationalDrugUse'] = this._recreationalDrugUse;
    data['stressManagement'] = this._stressManagement;
    data['mentalHealthHistory'] = this._mentalHealthHistory;
    data['familyHistory'] = this._familyHistory;
    data['sleepSchedule'] = this._sleepSchedule;
    data['screenTime'] = this._screenTime;
    data['occupation'] = this._occupation;
    data['occupationalExposures'] = this._occupationalExposures;
    data['travelHistory'] = this._travelHistory;
    data['menstrualHistory'] = this._menstrualHistory;
    data['additionalNotes'] = this._additionalNotes;
    return data;
  }
}
