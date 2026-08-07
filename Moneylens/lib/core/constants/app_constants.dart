class AppConstants {
  AppConstants._();

  static const String appName = 'MoneyLens';
  static const String appVersion = '1.0.0';

  // Database
  static const String dbName = 'moneylens.db';
  static const int dbVersion = 1;

  // Tables
  static const String tableTransactions = 'transactions';
  static const String tableMerchants = 'merchants';
  static const String tableSettings = 'settings';

  // Settings Keys
  static const String keyLastScanTime = 'last_scan_time';
  static const String keyOnboardingDone = 'onboarding_done';
  static const String keyTotalScanned = 'total_scanned';

  // Performance targets (ms)
  static const int dashboardLoadTarget = 500;
  static const int searchLoadTarget = 200;
}
