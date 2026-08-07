class CategoryConstants {
  CategoryConstants._();

  static const String food = 'Food';
  static const String travel = 'Travel';
  static const String shopping = 'Shopping';
  static const String bills = 'Bills';
  static const String entertainment = 'Entertainment';
  static const String medical = 'Medical';
  static const String fuel = 'Fuel';
  static const String grocery = 'Grocery';
  static const String salary = 'Salary';
  static const String transfer = 'Transfer';
  static const String others = 'Others';

  static const List<String> all = [
    food,
    travel,
    shopping,
    bills,
    entertainment,
    medical,
    fuel,
    grocery,
    salary,
    transfer,
    others,
  ];

  /// Default merchant → category mapping
  static const Map<String, String> merchantMap = {
    // Food
    'swiggy': food,
    'zomato': food,
    'dominos': food,
    'domino': food,
    'mcdonalds': food,
    'mcdonald': food,
    'kfc': food,
    'subway': food,
    'dunkin': food,
    'starbucks': food,
    'barbeque': food,

    // Grocery
    'blinkit': grocery,
    'zepto': grocery,
    'bigbasket': grocery,
    'big basket': grocery,
    'dmart': grocery,
    'jiomart': grocery,
    'grofers': grocery,
    'nature basket': grocery,

    // Travel
    'uber': travel,
    'ola': travel,
    'rapido': travel,
    'namma metro': travel,
    'irctc': travel,
    'makemytrip': travel,
    'goibibo': travel,
    'redbus': travel,
    'yatra': travel,

    // Shopping
    'amazon': shopping,
    'flipkart': shopping,
    'myntra': shopping,
    'ajio': shopping,
    'meesho': shopping,
    'nykaa': shopping,
    'snapdeal': shopping,
    'tatacliq': shopping,

    // Bills & Utilities
    'airtel': bills,
    'jio': bills,
    'vi ': bills,
    'vodafone': bills,
    'bescom': bills,
    'bbmp': bills,
    'bwssb': bills,
    'tata power': bills,
    'adani': bills,

    // Entertainment
    'netflix': entertainment,
    'spotify': entertainment,
    'bookmyshow': entertainment,
    'hotstar': entertainment,
    'disney': entertainment,
    'prime video': entertainment,
    'youtube': entertainment,
    'zee5': entertainment,
    'sonyliv': entertainment,

    // Medical
    'apollo': medical,
    'pharmaeasy': medical,
    'netmeds': medical,
    'medplus': medical,
    '1mg': medical,
    'practo': medical,

    // Fuel
    'hp petrol': fuel,
    'indian oil': fuel,
    'bharat petroleum': fuel,
    'iocl': fuel,
    'hpcl': fuel,
    'bpcl': fuel,
    'petrol': fuel,
    'shell': fuel,

    // Salary
    'salary': salary,
    'payroll': salary,
  };
}
