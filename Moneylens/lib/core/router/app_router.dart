import 'package:flutter/material.dart';
import 'package:moneylens/features/dashboard/presentation/home_screen.dart';
import 'package:moneylens/features/transactions/presentation/transactions_screen.dart';
import 'package:moneylens/features/categories/presentation/categories_screen.dart';
import 'package:moneylens/features/settings/settings_screen.dart';

class AppRoutes {
  static const String home = '/';
  static const String transactions = '/transactions';
  static const String categories = '/categories';
  static const String settings = '/settings';
  static const String permission = '/permission';
}

class AppRouter {
  static Route<dynamic> generateRoute(RouteSettings settings) {
    switch (settings.name) {
      case AppRoutes.home:
        return MaterialPageRoute(builder: (_) => const HomeScreen());
      case AppRoutes.transactions:
        return MaterialPageRoute(builder: (_) => const TransactionsScreen());
      case AppRoutes.categories:
        return MaterialPageRoute(builder: (_) => const CategoriesScreen());
      case AppRoutes.settings:
        return MaterialPageRoute(builder: (_) => const SettingsScreen());
      default:
        return MaterialPageRoute(
          builder: (_) => const Scaffold(
            body: Center(child: Text('Page not found')),
          ),
        );
    }
  }
}
